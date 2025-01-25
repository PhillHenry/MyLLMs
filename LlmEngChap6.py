from unsloth import PatchDPOTrainer

from config import SAVED_MODEL

PatchDPOTrainer()

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOConfig, DPOTrainer
from accelerate import init_empty_weights


class MyLlamaModel:
    max_seq_length = 512
    model_name="mlabonne/TwinLlama-3.1-8B"
    model_path = f"{SAVED_MODEL}/model_{model_name}"
    tokenizer_path = f"{SAVED_MODEL}/tokenizer_{model_name}"

    def get_model_tokenizer(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True, # "You can activate QLoRA by setting load_in_4bit to True"  LLMEngineering, p251
            # quantization_config=self.bnb_config, # helped with memory but caused non-zero probabilities when demoed
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
        model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.tokenizer_path)

    def load_prepared_dataset(self, EOS_TOKEN):
        alpaca_template = """Below is an instruction that describes a task.
        Write a response that appropriately completes the request.
        ### Instruction:
        {}
        ### Response:
        """

        def format_samples(example):
            example["prompt"] = alpaca_template.format(example["prompt"])
            example["chosen"] = example['chosen'] + EOS_TOKEN
            example["rejected"] = example['rejected'] + EOS_TOKEN
            return {"prompt": example["prompt"], "chosen":
                example["chosen"], "rejected": example["rejected"]}

        dataset = load_dataset("mlabonne/llmtwin-dpo", split="train")
        dataset = dataset.map(format_samples)
        dataset = dataset.train_test_split(test_size=0.05)
        return dataset


if __name__ == "__main__":
    my_model = MyLlamaModel()
    my_model.train_and_save()