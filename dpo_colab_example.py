# From https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing#scrollTo=E8-BWi7MzkRz
import comet_ml
from datasets import load_dataset
from unsloth import PatchDPOTrainer

from config import SAVED_MODEL
from preparation import get_datasets, to_dpo_format, apply_chat_template

PatchDPOTrainer()

from unsloth import FastLanguageModel

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


def main(tokenization_fn):
    model_name = "unsloth/zephyr-sft-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name =model_name, # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    raw_datasets = tokenization_fn(tokenizer)

    print_sample(raw_datasets, "train")
    print_sample(raw_datasets, "test")

    lora_alpha = 32
    learning_rate = 2e-6
    r = 32
    num_epochs = 2
    beta = 0.5

    model = FastLanguageModel.get_peft_model(
        model,
        r =r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha =lora_alpha,
        lora_dropout = 0, # Currently only supports dropout = 0
        bias = "none",    # Currently only supports bias = "none"
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # One must patch the DPO Trainer first!
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()

    from trl import DPOTrainer, DPOConfig
    from unsloth import is_bfloat16_supported

    eval_strategy = "steps"
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        eval_dataset=raw_datasets["test"],
        args = DPOConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_ratio = 0.1,
            num_train_epochs =num_epochs,
            learning_rate = learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            eval_strategy=eval_strategy, # needs an eval_dataset if using "steps"
            eval_steps=0.2,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs",
            report_to = "comet_ml", # Use this for WandB etc
        ),
        beta = beta,
        train_dataset = raw_datasets["train"],
        # eval_dataset = raw_datasets["test"],
        tokenizer = tokenizer,
        max_length = 1024,
        max_prompt_length = 1024,
    )

    dpo_trainer.train()

    model.save_pretrained_merged(f"{SAVED_MODEL}/r{r}_loraAlpha{lora_alpha}_epochs{num_epochs}_lr{learning_rate}_beta{beta}_eval_strategy{eval_strategy}/{model_name}", tokenizer=tokenizer, save_method="lora") # merged_4bit_forced


def print_sample(raw_datasets, train_or_test):
    import pprint
    print(f"\nsample {train_or_test}")
    row = raw_datasets[train_or_test][8]
    pprint.pprint(row["prompt"])
    pprint.pprint(row["chosen"])
    pprint.pprint(row["rejected"])


def ultrafeedback_tokenize_fn():
    data_model = "HuggingFaceH4/ultrafeedback_binarized"
    def do_tokenization(tokenizer):
        raw_datasets = get_datasets(
            {data_model: 0.005},  # 0.5% sampled
            splits=["train_prefs", "test_prefs"],
        )
        raw_datasets = to_dpo_format(raw_datasets, tokenizer)
        # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
        for split in ["train", "test"]:
            raw_datasets[split] = raw_datasets[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )
        return raw_datasets

    return do_tokenization

def labonne_tokenize_fn():
    def do_tokenization(tokenizer):
        dataset = load_dataset("mlabonne/llmtwin-dpo", split="train")
        def format_samples(example):
            actual_prompt = example['prompt']
            chosen = [{"content": actual_prompt, "role": "user"}, {"content": example['chosen'], "role": "assistant"}]
            messages = [{"content": actual_prompt, "role": "user"}, {"content": example['chosen'], "role": "assistant"}]
            rejected = [{"content": actual_prompt, "role": "user"}, {"content": example['rejected'], "role": "assistant"}]
            return {"prompt": actual_prompt, "chosen": chosen, "rejected": rejected, "messages": messages}

        dataset = dataset.map(format_samples)
        dataset = dataset.train_test_split(test_size=0.05)
        column_names = list(dataset["train"].features)
        dataset = dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
            num_proc=12,
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template",
        )
        for split in ["train", "test"]:
            dataset[split] = dataset[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )
        return dataset
    return do_tokenization

if __name__ == "__main__":
    main(labonne_tokenize_fn())
