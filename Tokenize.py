from datasets import load_dataset
from transformers import AutoTokenizer

from config import model_name

INPUT_TOKENIZED_DATASET = "input/tokenized_dataset"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Load a dataset
dataset = load_dataset("imdb")  # Replace "imdb" with your dataset

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.save_to_disk(INPUT_TOKENIZED_DATASET)