from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM

from config import SAVED_MODEL


def generate_text(model, tokenizer):
    # Input text
    prompt = "Tell me about Skyfall"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    output_tokens = model.generate(
        inputs["input_ids"],
        max_length=50,  # Limit the length of the generated text
        num_return_sequences=1,  # Generate a single sequence
        do_sample=True,  # Enable sampling for diverse outputs
        top_k=50,  # Limit sampling to top-k tokens
    )

    # Decode the generated tokens
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL)
    model = AutoModelForCausalLM.from_pretrained(SAVED_MODEL)
    generate_text(model, tokenizer)