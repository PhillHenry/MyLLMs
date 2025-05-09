from transformers import pipeline


def next_token(model):
    unmasker = pipeline('fill-mask', model=model)
    print(unmasker("L97809 W6199XS S31602A B2689 M130 S59199P I517 [MASK]"))


def play_with_bert():
    # a pre-trained model from HuggingFace
    model_name = "bert-pretrained"
    next_token(model_name)


if __name__ == "__main__":
    play_with_bert()