import pandas as pd
from transformers import BertTokenizerFast

from datasets import Dataset
from BertConfig import MY_VOCAB
from BertConfig import MY_CORPUS, MY_RESULTS

PREFIX = "200_"
LABEL = "labels"
TEXT_COL = "SNOMED"


def get_data_set() -> pd.DataFrame:
    results = pd.read_csv(PREFIX + MY_RESULTS, header=None, names=[LABEL])
    snomeds = pd.read_csv(PREFIX + MY_CORPUS, header=None, names=[TEXT_COL])

    assert len(snomeds) == len(results)

    df = pd.concat([snomeds, results], axis=1)
    df[LABEL] = df[LABEL].astype(int)
    print(df[LABEL].value_counts())
    df = Dataset.from_pandas(df)
    return df

def tokenize_dataset(df: pd.DataFrame, tokenizer: BertTokenizerFast):
    print("Tokenizing....")

    def tokenize(batch):
        tokens = tokenizer(batch[TEXT_COL], padding="max_length", truncation=True, max_length=200)
        print(len(tokens["input_ids"][0]))
        tokens[LABEL] = batch[LABEL]
        return tokens

    tokenized_dataset = df.map(tokenize, batched=True, remove_columns=[TEXT_COL],
                               load_from_cache_file=False)
    return tokenized_dataset