import torch
import numpy as np

from collections import Counter

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ValueError('This example requires huggingface-hub to load word vectors from the Hugging Face hub. '
                     'Please install huggingface-hub to run this example: '
                     'https://pypi.org/project/huggingface-hub/')

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from small_text import PytorchTextClassificationDataset

from examplecode.data.corpus_twenty_news import get_twenty_newsgroups_corpus


def get_train_test():
    return get_twenty_newsgroups_corpus(categories=['rec.sport.baseball', 'sci.med', 'rec.autos'])


def load_pretrained_word_vectors():
    vectors_path = hf_hub_download(repo_id='small-text/word2vec-google-news-300',
                                   filename='vectors.bin.npz')
    pretrained_vectors = np.load(vectors_path)

    words_path = hf_hub_download(repo_id='small-text/word2vec-google-news-300',
                                 filename='vocab.txt')
    words = _read_words(words_path)

    return words, pretrained_vectors['vectors']


def _read_words(words_file):
    words = []

    with open(words_file, 'r') as f:
        for line in f.readlines():
            words.append(line.strip())

    return words


def preprocess_data(train, test, words, pretrained_vectors,
                    vocab_size=200_000, unk_token='[UNK]', pad_token='[PAD]'):
    num_special_tokens = 2
    vocab = {
        words[i-num_special_tokens]: i
        for i in range(num_special_tokens, vocab_size)
    }

    vocab[unk_token] = 0  # [UNK]
    vocab[pad_token] = 1  # [PAD]

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])

    ds_train = PytorchTextClassificationDataset.from_arrays(train.data, train.target, tokenizer)
    ds_test = PytorchTextClassificationDataset.from_arrays(test.data, test.target, tokenizer)

    pretrained_vectors = _build_embeddings(ds_train.x, tokenizer, vocab, pretrained_vectors)

    return ds_train, ds_test, tokenizer, pretrained_vectors


def _build_embeddings(texts, tokenizer, vocab, pretrained_vectors, min_freq=1, num_special_tokens=2):

    vectors = [
        np.zeros(pretrained_vectors.shape[1])
        for _ in range(num_special_tokens)
    ]
    vectors += [
        pretrained_vectors[i]
        if tokenizer.id_to_token(i) in vocab
        else np.zeros(pretrained_vectors.shape[1])
        for i in range(num_special_tokens, len(vocab))
    ]

    token_id_list = [text.cpu().numpy().tolist() for text in texts]
    word_frequencies = Counter([token for tokens in token_id_list for token in tokens])
    for i in range(num_special_tokens, len(vocab)):
        is_in_vocab = tokenizer.id_to_token(i) in vocab
        if not is_in_vocab and word_frequencies[tokenizer.id_to_token(i)] >= min_freq:
            vectors[i] = np.random.uniform(-0.25, 0.25, pretrained_vectors.shape[1])

    return torch.as_tensor(np.stack(vectors))
