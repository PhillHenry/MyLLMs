from transformers import BertTokenizerFast, BertForSequenceClassification

from transformers import BertTokenizerFast, BertForSequenceClassification

from BertConfig import MY_VOCAB
from BertUtils import tokenize_dataset, get_data_set, get_data_frame, MODEL_FILE_NAME, LABEL, TEXT_COL
import torch

tokenizer = BertTokenizerFast.from_pretrained(MY_VOCAB)
tokenized_dataset = tokenize_dataset(get_data_set(), tokenizer)
model = BertForSequenceClassification.from_pretrained(MODEL_FILE_NAME)

df = get_data_frame()
print(df)

# Set model to evaluation mode
model.eval()

def make_prediction(texts):
    assert len(texts) > 0
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Print results
    for text, pred in zip(texts, predictions):
        print(f"Text: {text}")
        print(f"Predicted class: {pred.item()}")
        print("---")

diabetes = df[df[LABEL] == 1]
samples = diabetes[TEXT_COL][:10]
samples = samples.tolist()
print(samples)
make_prediction(samples)
