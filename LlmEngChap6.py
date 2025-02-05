import comet_ml
from unsloth import PatchDPOTrainer
from accelerate import Accelerator
from config import SAVED_MODEL

PatchDPOTrainer()

import torch
from transformers import TextStreamer, AutoTokenizer
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOConfig, DPOTrainer
from accelerate import init_empty_weights


class MyLlamaModel:
    max_seq_length = 256
    NUM_TRAIN_EPOCHS = 3
    beta = 0.5
    LOAD_IN_4BIT = False
    device_map = "auto"
    save_method = "lora"  # merged_X just means the whole model is saved, not just the transformer
    lora_dropout = 0.
    lora_alpha = 32
    learning_rate=2e-6
    r = 32
    base_output_dir = f"{SAVED_MODEL}/{max_seq_length}maxSeqLen_{NUM_TRAIN_EPOCHS}Epochs_{device_map}devmap_4Bit{LOAD_IN_4BIT}_{save_method}_beta{beta}_loraDropout{lora_dropout}_r{r}_lora_alpha{lora_alpha}_lr{learning_rate}/"

    def __init__(self):
        self.model_name="unsloth/Llama-3.2-3B-Instruct"  # "unsloth/Llama-3.1-Storm-8B-bnb-4bit"
        self.model_path = f"{self.base_output_dir}/{self.model_name}"

    def get_model_tokenizer(self, model_name: str):
        print(f"Using model {model_name}")
        self.model_name = model_name
        self.model_path = f"{self.base_output_dir}/{model_name}"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            # max_seq_length=self.max_seq_length,
            load_in_4bit=self.LOAD_IN_4BIT, # "You can activate QLoRA by setting load_in_4bit to True"  LLMEngineering, p251
            # quantization_config=bnb_config, # helped with memory but caused non-zero probabilities when demoed
            # # device_map=self.device_map, # try this
            # trust_remote_code=True,
        )
        return model, tokenizer

    def train_and_save(self):
        model, tokenizer = self.get_model_tokenizer(self.model_name)
        with init_empty_weights():
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            )
            torch.nn.Module.to_empty(model, device=torch.device("cuda"))  # this eliminates 'NotImplementedError: Cannot copy out of meta tensor'
            accelerator = Accelerator(mixed_precision="bf16", cpu=True)  # Enable mixed precision for memory efficiency
            device = accelerator.device
            # model.to(device)
            # optimizer = AdamW(params=model.parameters(), lr=3e-2)

            # Move the model to the appropriate device
            model = accelerator.prepare(model)
            self.do_dpo(model, tokenizer)

    def do_dpo(self, model, tokenizer):
        dataset = self.load_prepared_dataset(tokenizer.eos_token)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            beta=self.beta,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=self.max_seq_length // 2,
            max_prompt_length=self.max_seq_length // 2,
            args=DPOConfig(
                learning_rate=self.learning_rate,
                lr_scheduler_type="linear",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                num_train_epochs=self.NUM_TRAIN_EPOCHS,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
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
        model.save_pretrained_merged(self.model_path, tokenizer=tokenizer, save_method=self.save_method) # merged_4bit_forced
        # generate_text_using(model, tokenizer)


    @staticmethod
    def load_prepared_dataset(eos_token):
        template = (
        "<|begin_of_text|>\n"
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\n{prompt}\n"
        "<|assistant|>\n{chosen_response}\n"
        "<|assistant_rejected|>\n{rejected_response}\n"
        "<|end_of_text|>"
    )

        def format_samples(example):
            example["prompt"] = template.format(prompt=example["prompt"], chosen_response=example["chosen"], rejected_response=example["rejected"])
            example["chosen"] = example['chosen'] + eos_token
            example["rejected"] = example['rejected'] + eos_token
            return {"prompt": example["prompt"], "chosen":
                example["chosen"], "rejected": example["rejected"]}

        dataset = load_dataset("mlabonne/llmtwin-dpo", split="train")
        dataset = dataset.map(format_samples)
        dataset = dataset.train_test_split(test_size=0.05)
        return dataset


if __name__ == "__main__":
    my_model = MyLlamaModel()
    my_model.train_and_save()