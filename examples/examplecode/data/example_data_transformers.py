from small_text import TransformersDataset


def preprocess_data(tokenizer, texts, labels, max_length=500):
    return TransformersDataset.from_arrays(texts, labels, tokenizer, max_length=max_length)
