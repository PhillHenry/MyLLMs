from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import os
from datasets import load_dataset
from pathlib import Path

# path = str(Path("my_corpus.txt").resolve())
# dataset = load_dataset("text", data_files={"train": path})

# ========= 1. Train or load a tokenizer =========
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ========= 2. Load and tokenize your text dataset =========
dataset = load_dataset("text", data_files={"train": "my_training_data.txt"})

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

tokenized = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# ========= 3. Group into fixed-length chunks =========
block_size = 128

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

lm_dataset = tokenized.map(group_texts, batched=True)

# ========= 4. Create the model config and model =========
size = 12 * 20
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=size,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=2,
    hidden_size=size,
    intermediate_size=2048,
)
model = BertForMaskedLM(config)

# ========= 5. MLM data collator =========
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# ========= 6. Training arguments =========
training_args = TrainingArguments(
    output_dir="./bert-pretrained",
    overwrite_output_dir=True,
    # evaluation_strategy="no",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    fp16=True,
)

# ========= 7. Trainer =========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# ========= 8. Save final model and tokenizer =========
model.save_pretrained("./bert-pretrained")
tokenizer.save_pretrained("./bert-pretrained")
