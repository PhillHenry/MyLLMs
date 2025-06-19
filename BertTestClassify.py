import comet_ml
import pandas as pd

from datasets import Dataset
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizerFast, TrainingArguments, BertForSequenceClassification, Trainer

from BertConfig import MY_CORPUS, MY_RESULTS, bert_config, group_texts, MY_VOCAB
from BertUtils import tokenize_dataset

tokenizer = BertTokenizerFast.from_pretrained(MY_VOCAB)
tokenized_dataset = tokenize_dataset(df, tokenizer)