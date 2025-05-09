# MyLLMs
Me being silly with LLMs and various open source ML tools

# BERT

`BertPlayground.py` uses BERT to predict the next token in a sequence of ICD10 clinical codes.

`CreateICD10Sentances.py` create fake ICD10 codes. We can pretend that each line is a different patient.
Note that it needs to point at an Excel spreadsheet of codes where `CODE` is a column. 
I suggest using this [file](https://www.cms.gov/files/document/valid-icd-10-list.xlsx).

`BertPreTrainTokenizer.py` Creates the tokenizer for this corpus. 
You'll need to put your corpus file in this directory.

And `BertPreTrain.py` actually trains BERT from scratch using the tokenizer and corpus above.