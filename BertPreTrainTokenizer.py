from tokenizers.implementations import BertWordPieceTokenizer

from BertConfig import MY_VOCAB, MY_CORPUS
from utils import ensure_unique_dir

# 1. Train tokenizer on your corpus
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True
)

tokenizer.train(
    files=MY_CORPUS,
    vocab_size=2**17,
    min_frequency=2,
    limit_alphabet=1000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 2. Save it
ensure_unique_dir(MY_VOCAB)
tokenizer.save_model(MY_VOCAB)
tokenizer.save(f"{MY_VOCAB}/tokenizer.json")