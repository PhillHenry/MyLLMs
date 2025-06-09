from datasets import load_dataset
from transformers import (
    BertTokenizerFast,BertForSequenceClassification
)

from BertConfig import MY_CORPUS, MY_VOCAB, MY_RESULTS, bert_config
from torch.utils.data import DataLoader



# ========= 1. Train or load a tokenizer =========
tokenizer = BertTokenizerFast.from_pretrained(MY_VOCAB)
config = bert_config(tokenizer)
config.num_labels = 2
model = BertForSequenceClassification(config)


# ========= 2. Load and tokenize your text dataset =========
dataset = load_dataset("text", data_files={"train": MY_CORPUS, "labels": MY_RESULTS})

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

train_loader = DataLoader(tokenized_dataset['train'], batch_size=8, shuffle=True)
val_loader = DataLoader(tokenized_dataset['test'], batch_size=8)

#tokenized_dataset.save_to_disk()

model.train()