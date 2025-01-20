from peft import LoraConfig, PeftModel
from unsloth import PatchDPOTrainer

from config import SAVED_MODEL

PatchDPOTrainer()

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer, BitsAndBytesConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOConfig, DPOTrainer
from accelerate import init_empty_weights
import peft

max_seq_length = 512
model_name="mlabonne/TwinLlama-3.1-8B"
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

with init_empty_weights():
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    )

    # model.to(torch.device("cuda"))
    torch.nn.Module.to_empty(model, device=torch.device("cuda"))  # this eliminates 'NotImplementedError: Cannot copy out of meta tensor'

alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{}
### Response:
"""
EOS_TOKEN = tokenizer.eos_token
def format_samples(example):
    example["prompt"] = alpaca_template.format(example["prompt"])
    example["chosen"] = example['chosen'] + EOS_TOKEN
    example["rejected"] = example['rejected'] + EOS_TOKEN
    return {"prompt": example["prompt"], "chosen":
        example["chosen"], "rejected": example["rejected"]}

dataset = load_dataset("mlabonne/llmtwin-dpo", split="train")
dataset = dataset.map(format_samples)
dataset = dataset.train_test_split(test_size=0.05)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    beta=0.5,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_length=max_seq_length//2,
    max_prompt_length=max_seq_length//2,
    max_completion_length=max_seq_length//2,
    args=DPOConfig(
        learning_rate=2e-6,
        lr_scheduler_type="linear",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        report_to="comet_ml",
        seed=0,
        ),
    )
trainer.train()
model.save_pretrained(f"{SAVED_MODEL}/model_{model_name}")
tokenizer.save_pretrained(f"{SAVED_MODEL}/tokenizer_{model_name}")