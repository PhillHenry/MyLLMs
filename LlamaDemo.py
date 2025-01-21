from accelerate import init_empty_weights
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from transformers import AutoModelForCausalLM
from unsloth import FastLanguageModel

from LlmEngChap6 import MyLlamaModel
from Tokenize import INPUT_TOKENIZED_DATASET, tokenize
from config import SAVED_MODEL, MODEL_NAME, TOKENIZER_PATH
import torch

model_name = "mlabonne/TwinLlama-3.1-8B"
def basse_model_text():
    max_seq_length = 512
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        quantization_config=bnb_config,
    )
    inputs = tokenizer(["Is the James Bond film Skyfall good?"], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    FastLanguageModel.for_inference(model)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512, use_cache=True)


def transfer_model_text():
    model, tokenizer = FastLanguageModel.from_pretrained(f"{SAVED_MODEL}/model_{model_name}",
                                                         quantization_config=MyLlamaModel.bnb_config,
                                                         max_seq_length=MyLlamaModel.max_seq_length)
    model.cuda()
    FastLanguageModel.for_inference(model)
    alpaca_template = """Below is an instruction that describes a task.
    Write a response that appropriately completes the request.
    ### Instruction:
    {}
    ### Response:
    """
    message = alpaca_template.format("""Write a paragraph to introduce
    supervised
    fine - tuning.
    """, "")
    inputs = tokenizer([message], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs,streamer=text_streamer, max_new_tokens=MyLlamaModel.max_seq_length, use_cache=True)


if __name__ == "__main__":
    basse_model_text()
    # transfer_model_text()