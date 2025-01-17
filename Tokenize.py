from datasets import load_dataset
from transformers import GPT2Tokenizer

from config import MODEL_NAME

INPUT_TOKENIZED_DATASET = "./tokenized_gpt2_dataset"

def do_tokenization(model_name: str):
    tokenized_dataset = tokenize()
    tokenized_dataset.save_to_disk(INPUT_TOKENIZED_DATASET)
    return tokenized_dataset


def tokenize():
    # Load a dataset
    dataset = load_dataset("imdb")  # Replace "imdb" with your dataset
    tokenizer = get_tokenizer()
    # Tokenize the dataset
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'
    return tokenizer


if __name__ == "__main__":
    do_tokenization(MODEL_NAME)