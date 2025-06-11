import pandas as pd

from datasets import Dataset
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizerFast, TrainingArguments, BertForSequenceClassification, Trainer

from BertConfig import MY_CORPUS, MY_RESULTS, bert_config, group_texts, MY_VOCAB

PREFIX = "200_"
LABEL = "labels"
TEXT_COL = "SNOMED"

snomeds = pd.read_csv(PREFIX + MY_CORPUS, header=None, names=[TEXT_COL])
results = pd.read_csv(PREFIX + MY_RESULTS, header=None, names=[LABEL])

assert len(snomeds) == len(results)

df = pd.concat([snomeds, results], axis=1)
df = Dataset.from_pandas(df)

print("Training Tokenizer....")
tokenizer = BertWordPieceTokenizer(clean_text=True)
tokenizer.train_from_iterator(df[TEXT_COL])

tokenizer.save_model(MY_VOCAB)
tokenizer.save(f"{MY_VOCAB}/tokenizer.json")

print("Tokenizing....")

tokenizer = BertTokenizerFast.from_pretrained(MY_VOCAB)
def tokenize(batch):
    tokens = tokenizer(batch[TEXT_COL], padding="max_length", truncation=True, max_length=200)
    print(len(tokens["input_ids"][0]))
    tokens[LABEL] = batch[LABEL]
    return tokens

tokenized_dataset = df.map(tokenize, batched=True, remove_columns=[TEXT_COL], load_from_cache_file=False)

config = bert_config(tokenizer)
config.num_labels = 1
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


# block_size = 128
# def group_texts(examples):
#     concatenated = {k: sum(examples[k], []) for k in examples.keys() if not k == LABEL}
#     total_length = (len(concatenated["input_ids"]) // block_size) * block_size
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated.items()
#     }
#     return result
# lm_dataset = tokenized_dataset.map(group_texts, batched=True)



