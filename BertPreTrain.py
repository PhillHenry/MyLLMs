from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from BertConfig import SAVE_DIRECTORY, MY_CORPUS, MY_VOCAB
from utils import ensure_unique_dir

# ========= 1. Train or load a tokenizer =========
tokenizer = BertTokenizerFast.from_pretrained(MY_VOCAB)

# ========= 2. Load and tokenize your text dataset =========
dataset = load_dataset("text", data_files={"train": MY_CORPUS})

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
num_attention_heads = 12
size = num_attention_heads * 20
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=size,
    num_attention_heads=num_attention_heads,
    num_hidden_layers=6,
    type_vocab_size=2,
    hidden_size=size,
    intermediate_size=2048,
)
model = BertForMaskedLM(config)

# ========= 5. MLM data collator =========
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.1
)

# ========= 6. Training arguments =========
training_args = TrainingArguments(
    output_dir="./bert-pretrained",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=1,
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
ensure_unique_dir(SAVE_DIRECTORY)
model.save_pretrained(SAVE_DIRECTORY)
tokenizer.save_pretrained(SAVE_DIRECTORY)
