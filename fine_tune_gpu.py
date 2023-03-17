from datasets import Dataset
import os
import pandas as pd
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

is_cuda_available = torch.cuda.is_available()
device = "cuda:0" if is_cuda_available else "cpu"

# Tokenize the text using the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
output_dir = "./output"

def dataset_from_csv(location=None):
    global tokenizer
    if location is None:
        raise ValueError("Please provide the location of the CSV file.")
    dataset = Dataset.from_pandas(pd.read_csv(location))
    dataset = dataset.filter(lambda example: example["language"] == "python")
    def combine_docstring_and_code(example):
        example["text"] = example["docstring"] + " " + example["code"]
        return example
    dataset = dataset.map(combine_docstring_and_code)
    def tokenize(example):
        example["tokens"] = example["docstring_tokens"] + example["code_tokens"]
        example["input_ids"] = tokenizer.encode(example["tokens"], padding="max_length", truncation=True)
        example["attention_mask"] = [float(i>0) for i in example["input_ids"]]
        return example
    dataset = dataset.map(tokenize)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset

train_dataset = dataset_from_csv('datasets/codesearchnet_train_py_small.csv')
eval_dataset = dataset_from_csv('datasets/codesearchnet_valid_py_small.csv')

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=21,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=2e-5,
    optim="adamw_torch",
    fp16=is_cuda_available,
    use_mps_device=False,
    logging_steps=100,
    dataloader_num_workers=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    prediction_loss_only=True
)

def data_collator(data):
    input_ids = torch.stack([f["input_ids"] for f in data])
    attention_mask = torch.stack([f["attention_mask"] for f in data])
    labels = torch.stack([f["input_ids"] for f in data])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

resume_from_checkpoint = False
if os.path.isdir(output_dir):
    for element in os.listdir(output_dir):
        if re.match("(checkpoint-\d+)", element):
            resume_from_checkpoint=True
            break
        
trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.save_model()

prompt = "This function returns the sum of two numbers"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=1024, num_return_sequences=1)
generated_code = tokenizer.decode(output[0][0], skip_special_tokens=True)
print(generated_code)
