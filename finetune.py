import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Tokenize the text using the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2")

def dataset_from_csv(location=None):
    df = pd.read_csv(location)
    dataset = Dataset.from_pandas(df)
    # Filter the dataset to only include Python code
    dataset = dataset.filter(lambda example: example["language"] == "python")
    # Combine the docstring and code into a single string
    dataset = dataset.map(lambda example: {"text": example["docstring"] + " " + example["code"]})
    # Tokenize the text using the GPT-2 tokenizer
    dataset = dataset.map(lambda example: {"tokens": example["docstring_tokens"] + example["code_tokens"]})
    dataset = dataset.map(lambda example: {"input_ids": tokenizer.encode(example["tokens"], padding="max_length", truncation=True)})
    dataset = dataset.map(lambda example: {"attention_mask": [float(i>0) for i in example["input_ids"]]})
    # Set the max length to 1024
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset

train_dataset = dataset_from_csv('datasets/codesearchnet_train_py_small.csv')
eval_dataset = dataset_from_csv('datasets/codesearchnet_valid_py_small.csv')


training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=2e-5,
    optim="adamw_torch",
    # fp16=True,
    use_mps_device=torch.backends.mps.is_available(),
    logging_steps=100,
    dataloader_num_workers=0,
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
trainer.train()

prompt = "This function returns the sum of two numbers"
inputs = tokenizer(prompt, return_tensors="pt")
model = GPT2LMHeadModel.from_pretrained("gpt2")
output = model.generate(**inputs, max_length=1024, num_return_sequences=1)
generated_code = tokenizer.decode(output[0][0], skip_special_tokens=True)
print(generated_code)
