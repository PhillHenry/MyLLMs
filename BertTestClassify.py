import pandas as pd
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

    return list(map(lambda x: x.item(), predictions))


def test_with_label(label: int) -> pd.DataFrame:
    cohort = df[df[LABEL] == label]
    samples = cohort[TEXT_COL][:10]
    samples = samples.tolist()
    predictions = make_prediction(samples)
    print(f"Label {label}: accuracy = {len([x for x in predictions if x == label])} / {len(samples)}")
    return samples


def sense_check_sample(sample: [str], expected: int):
    print(f"Checking {len(sample)} samples have class {expected}")
    for codes in sample:
        is_diabetic = False
        for code in codes.split(" "):
            if code.startswith("E0"):
                is_diabetic = True
        if expected == 1:
            assert is_diabetic
        else:
            assert not is_diabetic

diabetics = test_with_label(1)
non_diabetics = test_with_label(0)

sense_check_sample(diabetics, 1)
sense_check_sample(non_diabetics, 0)

