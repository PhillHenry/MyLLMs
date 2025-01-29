import comet_ml
from transformers import BitsAndBytesConfig
from unsloth import PatchDPOTrainer
from accelerate import Accelerator
from config import SAVED_MODEL

PatchDPOTrainer()

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOConfig, DPOTrainer
from accelerate import init_empty_weights, cpu_offload
from torch.optim import AdamW



class MyLlamaModel:
    max_seq_length = 512
    model_name="unsloth/Llama-3.2-1B-Instruct"
    NUM_TRAIN_EPOCHS = 1
    LOAD_IN_4BIT = False
    device_map = "auto"
    save_method = "lora"
    base_output_dir = f"{SAVED_MODEL}/{max_seq_length}maxSeqLen_{NUM_TRAIN_EPOCHS}Epochs_{device_map}devmap_4Bit{LOAD_IN_4BIT}_{save_method}/"
    model_path = f"{base_output_dir}/{model_name}"

    def get_model_tokenizer(self):
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
        model, tokenizer = self.get_model_tokenizer()
        with init_empty_weights():
            model = FastLanguageModel.get_peft_model(
                model,
                r=32,
                lora_alpha=32,
                lora_dropout=0,
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
            beta=0.5,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=self.max_seq_length // 2,
            max_prompt_length=self.max_seq_length // 2,
            args=DPOConfig(
                learning_rate=2e-6,
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

    @staticmethod
    def load_prepared_dataset(eos_token):
        alpaca_template = """Below is an instruction that describes a task.
        Write a response that appropriately completes the request.
        ### Instruction:
        {}
        ### Response:
        """

        def format_samples(example):
            example["prompt"] = alpaca_template.format(example["prompt"])
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