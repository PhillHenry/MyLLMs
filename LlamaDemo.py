from accelerate import Accelerator
from transformers import TextStreamer, AutoTokenizer
from unsloth import FastLanguageModel

from LlmEngChap6 import MyLlamaModel, generate_text_using
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


if __name__ == "__main__":
    # Eg, python LlamaDemo.py unsloth/Llama-3.2-1B-Instruct
    if len(sys.argv) == 1:
        generate(MyLlamaModel())
    else:
        path = sys.argv[1]
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=path)
        generate_text_using(model, tokenizer)