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
