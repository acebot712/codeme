import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("codesearchnet")

# Filter the dataset to only include Python code
python_dataset = dataset["train"].filter(lambda example: example["language"] == "python")

# Combine the docstring and code into a single string
python_dataset = python_dataset.map(lambda example: {"text": example["docstring"] + " " + example["code"]})

# Tokenize the text using the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenized_dataset = python_dataset.map(lambda example: tokenizer(example["text"], padding="max_length", truncation=True), batched=True)

# Set the max length to 1024
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=100,
    dataloader_num_workers=4,
    evaluation_strategy="epoch",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {"input_ids": torch.stack([f["input_ids"] for f in data]),
                                "attention_mask": torch.stack([f["attention_mask"] for f in data]),
                                "labels": torch.stack([f["input_ids"] for f in data])},
    prediction_loss_only=True,
)
trainer.train()

prompt = "This function returns the sum of two numbers"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids=input_ids, max_length=1024, num_return_sequences=1)
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)

# fine tune a gptneo model from hugging face with pytorch using prompt and text pairs

#import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#instantiate tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

#define prompt and text pairs
prompt = "The cat sat on the"
text_pairs = [
    "mat. It was very comfortable.",
    "chair. It didn't like it very much."
]

#encode prompt and text pairs
encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
encoded_text_pairs = [tokenizer.encode(pair, add_special_tokens=False) for pair in text_pairs]

#create input tensors
input_ids = torch.tensor([encoded_prompt + pair for pair in encoded_text_pairs])

#fine tune the model
model.train()
model.fit(input_ids, labels=input_ids)

# calculate  Normalized Discounted Cumulative Gain from generated code token list and ground truth code tokens list

#importing libraries
import numpy as np

#ground truth code tokens list
ground_truth_code_tokens = ['if', 'else', 'for', 'while', 'break', 'continue']

#generated code token list
generated_code_tokens = ['if', 'else', 'for', 'while', 'break', 'continue', 'int', 'float']

#calculating normalized discounted cumulative gain
def ndcg(ground_truth_code_tokens, generated_code_tokens):
    #calculating ideal discounted cumulative gain
    idcg = 0
    for i in range(len(ground_truth_code_tokens)):
        idcg += (2**i - 1)/np.log2(i+2)
    
    #calculating discounted cumulative gain
    dcg = 0
    for i in range(len(generated_code_tokens)):
        if generated_code_tokens[i] in ground_truth_code_tokens:
            dcg += (2**i - 1)/np.log2(i+2)
    
    #calculating normalized discounted cumulative gain
    ndcg = dcg/idcg
    
    return ndcg
