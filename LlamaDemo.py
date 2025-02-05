from accelerate import Accelerator
from transformers import TextStreamer, AutoTokenizer
from unsloth import FastLanguageModel

from LlmEngChap6 import MyLlamaModel
import sys
import torch


def generate(model: MyLlamaModel):
    model_name = model.model_path
    print("Generating text for " + model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=model.LOAD_IN_4BIT,
    )
    # accelerator = Accelerator(mixed_precision="fp16", cpu=True)  # Enable mixed precision for memory efficiency
    # model = accelerator.prepare(model)
    # model, tokenizer = AutoTokenizer.from_pretrained(
    #     max_seq_length=MyLlamaModel.max_seq_length,
    #     pretrained_model_name_or_path=model_name
    # )
    generate_text_using(model, tokenizer)


def generate_text_using(model, tokenizer):
    print(f"Model of type {type(model)}, tokenizer of type {type(tokenizer)}")
    #"pt",  "tf",  "np", "jax", "mlx"
    inputs = tokenizer(["Who are the creators of the course that is under the 'Decoding ML' umbrella?"], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    FastLanguageModel.for_inference(model)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=MyLlamaModel.max_seq_length, use_cache=True)


if __name__ == "__main__":
    # Eg, python LlamaDemo.py mlabonne/TwinLlama-3.1-8B-DPO # the actual artifact
    if len(sys.argv) == 1:
        print("Using default model")
        generate(MyLlamaModel())
    else:
        path = sys.argv[1]
        print(f"using {path}")
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=path)
        generate_text_using(model, tokenizer)