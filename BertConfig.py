from transformers import (
    BertTokenizerFast,
    BertConfig,
)

SAVE_DIRECTORY = "./bert-pretrained"
MY_CORPUS = "my_training_data.txt"
MY_RESULTS = "my_training_results.txt"
MY_VOCAB = "./tokenizer-output"

def bert_config(tokenizer: BertTokenizerFast):
    num_attention_heads = 40
    size = num_attention_heads * 20
    return BertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=6,
        type_vocab_size=2,
        hidden_size=size,
        intermediate_size=2048,
    )

block_size = 128

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result