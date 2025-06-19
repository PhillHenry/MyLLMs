import comet_ml

from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizerFast, TrainingArguments, BertForSequenceClassification, \
    Trainer

from BertConfig import bert_config, MY_VOCAB
from BertUtils import TEXT_COL, tokenize_dataset, get_data_set, MODEL_FILE_NAME, TEST_FILE_NAME, \
    TRAIN_FILE_NAME

ds = get_data_set()

print("Training Tokenizer....")
tokenizer = BertWordPieceTokenizer(clean_text=True)
tokenizer.train_from_iterator(ds[TEXT_COL])

tokenizer.save_model(MY_VOCAB)
tokenizer.save(f"{MY_VOCAB}/tokenizer.json")

tokenizer = BertTokenizerFast.from_pretrained(MY_VOCAB)
tokenized_dataset = tokenize_dataset(ds, tokenizer, remove_columns=[])

config = bert_config(tokenizer)
config.num_labels = 2
model = BertForSequenceClassification(config)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir='./logs',
    report_to="comet_ml",
)

split_ds = tokenized_dataset.train_test_split(test_size=0.2)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

model.save_pretrained(MODEL_FILE_NAME)
print(f"Saved {MODEL_FILE_NAME}")

train_ds.save_to_disk(TRAIN_FILE_NAME)
print(f"Saved {TRAIN_FILE_NAME}")

test_ds.save_to_disk(TEST_FILE_NAME)
print(f"Saved {TEST_FILE_NAME}")