from accelerate import init_empty_weights
from datasets import DatasetDict
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TextStreamer, LlamaModel
from transformers import AutoModelForCausalLM
from unsloth import FastLanguageModel

from LlmEngChap6 import MyLlamaModel
from Tokenize import INPUT_TOKENIZED_DATASET, tokenize
from config import SAVED_MODEL, MODEL_NAME, TOKENIZER_PATH
import torch

def basse_model_text():
    generate(MyLlamaModel.model_name)

def generate(model_name: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MyLlamaModel.model_name,
        max_seq_length=MyLlamaModel.max_seq_length,
        load_in_4bit=True,
        # quantization_config=self.bnb_config,
    )
    generate_text_using(model, tokenizer)


def generate_text_using(model, tokenizer):
    inputs = tokenizer(["Is the James Bond film Skyfall good?"], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    FastLanguageModel.for_inference(model)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=MyLlamaModel.max_seq_length, use_cache=True)


def transfer_model_text():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MyLlamaModel.model_name,
        max_seq_length=MyLlamaModel.max_seq_length,
        load_in_4bit=True,
        # quantization_config=self.bnb_config,
    )
    model = PeftModel.from_pretrained(
        model=model,
        model_id=MyLlamaModel.model_path,
    )
    # tokenizer = AutoTokenizer.from_pretrained(MyLlamaModel.tokenizer_path)

    # model.cuda()
    generate_text_using(model, tokenizer)


if __name__ == "__main__":
    generate(MyLlamaModel.model_path)
    # basse_model_text()
    # transfer_model_text()