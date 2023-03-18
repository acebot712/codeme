import argparse
from datasets import Dataset
import os
import pandas as pd
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Define command line arguments</span>
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help="Device to use for training, either 'cpu' or 'cuda'")
parser.add_argument("--train_csv", type=str, help="Location of train CSV file")
parser.add_argument("--eval_csv", type=str, help="Location of eval CSV file")

# Parse command line arguments</span>
args = parser.parse_args()

# Set the device based on user input</span>
device = args.device
# Check if CUDA is available
is_cuda_available = torch.cuda.is_available()
if device == 'cuda' and not is_cuda_available:
    raise ValueError("Cuda device not available!")

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the pad token to the end of sentence token
tokenizer.pad_token = tokenizer.eos_token

# Initialize GPT-2 model and move it to the device
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Set output directory
output_dir = "./output"

# Function to read CSV file and return a dataset
def dataset_from_csv(location=None):
    """
    Reads a CSV file and returns a dataset

    Parameters:
    location (str): Path to the CSV file

    Returns:
    Dataset: A dataset object
    """
    global tokenizer

    # Raise an error if no location is provided
    if location is None:
        raise ValueError("Please provide the location of the CSV file.")

    # Read the CSV file and create a dataset from it
    dataset = Dataset.from_pandas(pd.read_csv(location))

    # Filter out all examples that are not Python code
    dataset = dataset.filter(lambda example: example["language"] == "python")

    # Combine the docstring and code into one string
    def combine_docstring_and_code(example):
        example["text"] = example["docstring"] + " " + example["code"]
        return example

    dataset = dataset.map(combine_docstring_and_code)

    # Tokenize the text using the GPT-2 tokenizer
    def tokenize(example):
        example["tokens"] = example["docstring_tokens"] + example["code_tokens"]
        example["input_ids"] = tokenizer.encode(example["tokens"], padding="max_length", truncation=True)
        example["attention_mask"] = [float(i>0) for i in example["input_ids"]]
        return example

    dataset = dataset.map(tokenize)

    # Set the format of the dataset to Torch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset

# Create datasets from the CSV files
train_dataset = dataset_from_csv(args.train_csv)
eval_dataset = dataset_from_csv(args.eval_csv)

# Set training arguments
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
    dataloader_num_workers=0,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    prediction_loss_only=True
)

# Function to collate data
def data_collator(data):
    input_ids = torch.stack([f["input_ids"] for f in data])
    attention_mask = torch.stack([f["attention_mask"] for f in data])
    labels = torch.stack([f["input_ids"] for f in data])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Check if there is a checkpoint in the output directory
resume_from_checkpoint = False
if os.path.isdir(output_dir):
    for element in os.listdir(output_dir):
        if re.match("(checkpoint-\d+)", element):
            resume_from_checkpoint=True
            break

# Train the model
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# Save the model
trainer.save_model()

# Generate code given a prompt
prompt = "This function returns the sum of two numbers"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=1024, num_return_sequences=1)
generated_code = tokenizer.decode(output[0][0], skip_special_tokens=True)
print(generated_code)