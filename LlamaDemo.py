from transformers import TextStreamer
from unsloth import FastLanguageModel

from LlmEngChap6 import MyLlamaModel


def basse_model_text():
    generate(MyLlamaModel.model_name)

def generate(model_name: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MyLlamaModel.model_name,
        max_seq_length=MyLlamaModel.max_seq_length,
        load_in_4bit=True,
    )
    generate_text_using(model, tokenizer)


def generate_text_using(model, tokenizer):
    alpaca_template = """Below is an instruction that describes a task.
        Write a response that appropriately completes the request.
        ### Instruction:
        {}
        ### Response:
        """
    message = alpaca_template.format("""Write a paragraph to introduce zero shot learning
        """, "")
    inputs = tokenizer([message], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    FastLanguageModel.for_inference(model)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=MyLlamaModel.max_seq_length, use_cache=True)


if __name__ == "__main__":
    generate(MyLlamaModel.model_path)
    # basse_model_text()