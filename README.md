# MyLLMs

Katas for LLMs etc.

# BERT

Run the scripts in this order:

1. `CreateICD10Sentances.py` create fake ICD10 codes. We can pretend that each line is a different patient.
Note that it needs to point at an Excel spreadsheet of codes where `CODE` is a column. 
I suggest using this [file](https://www.cms.gov/files/document/valid-icd-10-list.xlsx).

2. `BertPreTrainTokenizer.py` Creates the tokenizer for this corpus.

3. And `BertPreTrain.py` actually trains BERT from scratch using the tokenizer and corpus above.

4. `BertPlayground.py` uses BERT to predict the next token in a sequence of ICD10 clinical codes.
