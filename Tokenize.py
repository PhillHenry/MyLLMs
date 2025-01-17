from datasets import load_dataset
from transformers import AutoTokenizer, GPT2TokenizerFast, GPT2Tokenizer

from config import MODEL_NAME

INPUT_TOKENIZED_DATASET = "./tokenized_gpt2_dataset"

def do_tokenization(model_name: str):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'

    # Load a dataset
    dataset = load_dataset("imdb")  # Replace "imdb" with your dataset

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.save_to_disk(INPUT_TOKENIZED_DATASET)
    return tokenized_dataset

if __name__ == "__main__":
    do_tokenization(MODEL_NAME)