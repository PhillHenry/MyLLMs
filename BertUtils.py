import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast

from BertConfig import MY_CORPUS, MY_RESULTS, MY_VOCAB

PREFIX = "200_"
LABEL = "labels"
TEXT_COL = "SNOMED"
MODEL_FILE_NAME = f"{MY_VOCAB}/model"


def get_data_set() -> Dataset:
    df = get_data_frame()
    print(df[LABEL].value_counts())
    df = Dataset.from_pandas(df)
    return df


def get_data_frame() -> pd.DataFrame:
    results = pd.read_csv(PREFIX + MY_RESULTS, header=None, names=[LABEL])
    snomeds = pd.read_csv(PREFIX + MY_CORPUS, header=None, names=[TEXT_COL])
    assert len(snomeds) == len(results)
    df = pd.concat([snomeds, results], axis=1)
    df[LABEL] = df[LABEL].astype(int)
    return df


def tokenize_dataset(df: pd.DataFrame, tokenizer: BertTokenizerFast):
    print("Tokenizing....")

    def tokenize(batch):
        tokens = tokenizer(batch[TEXT_COL],
                           padding="max_length",
                           truncation=True,
                           # truncation_strategy="do_not_truncate",
                           max_length=240
                           )
        print(f"Length of input_ids = {len(tokens['input_ids'][0])}")
        tokens[LABEL] = batch[LABEL]
        return tokens

    tokenized_dataset = df.map(tokenize,
                               batched=True,
                               remove_columns=[TEXT_COL],
                               load_from_cache_file=False)
    return tokenized_dataset