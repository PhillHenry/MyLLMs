from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM

from Tokenize import INPUT_TOKENIZED_DATASET, tokenize
from config import SAVED_MODEL, MODEL_NAME, TOKENIZER_PATH


def generate_text(model, tokenizer):
    # Input text
    prompt = "Is the James Bond film Skyfall good?"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    output_tokens = model.generate(
        inputs["input_ids"],
        max_length=100,  # Limit the length of the generated text
        num_return_sequences=1,  # Generate a single sequence
        do_sample=True,  # Enable sampling for diverse outputs
        top_k=50,  # Limit sampling to top-k tokens
    )

    # Decode the generated tokens
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    print(f"Generated text: {generated_text}")


def basse_model_text():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    generate_text(model, tokenizer)


def transfer_model_text():
    # tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL)
    # tokenized_dataset = DatasetDict.load_from_disk(INPUT_TOKENIZED_DATASET)
    # t = tokenize()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    imdb_model = AutoModelForCausalLM.from_pretrained(SAVED_MODEL)
    generate_text(imdb_model, tokenizer)


if __name__ == "__main__":
    transfer_model_text()
    basse_model_text()