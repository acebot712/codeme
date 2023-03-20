import argparse
from datasets import Dataset
import os
import pandas as pd
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import time
import sys

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the pad token to the end of sentence token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained("./vm/")

# Initialize GPT-2 model and move it to the device
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.save_pretrained("./vm/")
