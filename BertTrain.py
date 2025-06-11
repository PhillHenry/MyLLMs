from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from BertConfig import SAVE_DIRECTORY, MY_CORPUS, MY_VOCAB, bert_config
from utils import ensure_unique_dir

# ========= 1. Train or load a tokenizer =========
tokenizer = BertTokenizerFast.from_pretrained(MY_VOCAB)

# ========= 2. Load and tokenize your text dataset =========
dataset = load_dataset("text", data_files={"train": MY_CORPUS})

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

lm_dataset = dataset.map(group_texts, batched=True)
