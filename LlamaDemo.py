from accelerate import Accelerator
from transformers import TextStreamer
from unsloth import FastLanguageModel

from LlmEngChap6 import MyLlamaModel
import sys


def basse_model_text():
    generate(MyLlamaModel.model_name)

def generate(model_name: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MyLlamaModel.model_name,
        max_seq_length=MyLlamaModel.max_seq_length,
        load_in_4bit=True,
    )
    accelerator = Accelerator(mixed_precision="fp16", cpu=True)  # Enable mixed precision for memory efficiency
    model = accelerator.prepare(model)
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
    if len(sys.argv) == 0:
        path = MyLlamaModel.model_path
    else:
        path = sys.argv[1]
    generate(path)
    # basse_model_text()