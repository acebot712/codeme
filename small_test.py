from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import nlpaug.augmenter.word as naw
from fine_tune_gpu import dataset_from_csv

def read_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read()

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large-ntp-py")

train_dataset = dataset_from_csv("./datasets/codesearchnet_train_py_small.csv", tokenizer=tokenizer)
eval_dataset = dataset_from_csv("./datasets/codesearchnet_valid_py_small.csv", tokenizer=tokenizer)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

text = read_from_file('sample.txt')

"""
aug = naw.RandomWordAug(action="delete", aug_p=0.3)
augmented_text = aug.augment(text)
print(augmented_text)

aug = naw.RandomWordAug(action="substitute", aug_p=0.3)
augmented_text = aug.augment(augmented_text)
print(augmented_text)
"""

input_ids = tokenizer(text, return_tensors="pt").input_ids
# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=128, num_beams=5, no_repeat_ngram_size=2)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# remove duplicates
generated_lines = set(generated_text.split("\n"))
generated_text = '\n'.join(generated_lines)

print(generated_text)
