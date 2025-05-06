import comet_ml

comet_ml.init(api_key="wfYwSiDlqTHVyTWHjAuT6qI0P")


from transformers import BertTokenizer, BertForSequenceClassification

from LlmEngChap6 import MyLlamaModel



model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # for binary classification


from datasets import load_dataset

dataset = load_dataset("imdb")  # or your own dataset
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")  # required by HF
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

from torch.utils.data import DataLoader

train_loader = DataLoader(tokenized_dataset['train'], batch_size=8, shuffle=True)
val_loader = DataLoader(tokenized_dataset['test'], batch_size=8)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir='./logs',
    report_to="comet_ml",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
)

trainer.train()

#model_path = f"{MyLlamaModel.base_output_dir}/{model_name}"
model_path = f"{MyLlamaModel.base_output_dir}/"
model.save_pretrained(model_path)
#model.save_pretrained_merged(model_path, tokenizer=tokenizer) # merged_4bit_forced
