from tokenizers.implementations import BertWordPieceTokenizer

MY_CORPUS = "my_training_data.txt"
MY_VOCAB = "./tokenizer-output"

# 1. Train tokenizer on your corpus
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True
)

tokenizer.train(
    files=MY_CORPUS,
    vocab_size=30522,  # standard BERT vocab size
    min_frequency=2,
    limit_alphabet=1000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 2. Save it
tokenizer.save_model(MY_VOCAB)
tokenizer.save(f"{MY_VOCAB}/tokenizer.json")