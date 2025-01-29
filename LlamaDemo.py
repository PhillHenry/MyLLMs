from accelerate import Accelerator
from transformers import TextStreamer, AutoTokenizer
from unsloth import FastLanguageModel

from LlmEngChap6 import MyLlamaModel
import sys
import torch


def generate(model: MyLlamaModel):
    print("Generating text for " + model.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model.model_name,
        load_in_4bit=model.LOAD_IN_4BIT,
        # dtype=torch.bfloat16
        # max_seq_length=MyLlamaModel.max_seq_length,
        # pretrained_model_name_or_path=model_name
        # load_in_4bit=True,
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
    inputs = tokenizer(["""Write a paragraph to introduce zero shot learning"""], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    FastLanguageModel.for_inference(model)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=MyLlamaModel.max_seq_length, use_cache=True)


if __name__ == "__main__":
    # Eg, python LlamaDemo.py unsloth/Llama-3.2-1B-Instruct
    if len(sys.argv) == 1:
        generate(MyLlamaModel())
    else:
        path = sys.argv[1]
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=path)
        generate_text_using(model, tokenizer)
    # basse_model_text()