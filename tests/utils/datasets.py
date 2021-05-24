import numpy as np

from scipy import sparse
from sklearn.datasets import fetch_20newsgroups

from active_learning.data.datasets import SklearnDataSet

try:
    import torch
    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from active_learning.integrations.transformers.datasets import TransformersDataset
except ImportError as e:
    pass

try:
    from transformers import AutoTokenizer
except ImportError as e:
    pass


def random_matrix_data(matrix_type, num_samples=100, num_dimension=40, num_labels=2):
    if matrix_type == 'dense':
        x = np.random.rand(num_samples, num_dimension)
    else:
        x = sparse.random(num_samples, num_dimension, density=0.15, format='csr')

    y = np.random.randint(0, high=num_labels, size=x.shape[0])
    return x, y


def random_sklearn_dataset(num_samples, vocab_size=60, num_classes=2):

    x = sparse.random(num_samples, vocab_size, density=0.15, format='csr')
    y = np.random.randint(0, high=num_classes, size=x.shape[0])

    return SklearnDataSet(x, y)


def trec_dataset():
    import torchtext

    try:
        from torchtext import data
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False, unk_token=None)
        train, test = torchtext.datasets.TREC.splits(text_field, label_field)
    except AttributeError:
        # torchtext >= 0.8.0
        from torchtext.legacy import data
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False, unk_token=None)
        train, test = torchtext.legacy.datasets.TREC.splits(text_field, label_field)

    text_field.build_vocab(train)
    label_field.build_vocab(train)

    return _dataset_to_text_classification_dataset(train), \
        _dataset_to_text_classification_dataset(test)


def _dataset_to_text_classification_dataset(dataset):
    import torch

    assert dataset.fields['text'].vocab.itos[0] == '<unk>'
    assert dataset.fields['text'].vocab.itos[1] == '<pad>'
    unk_token_idx = 1

    vocab = dataset.fields['text'].vocab

    data = [(torch.LongTensor([vocab.stoi[token] if token in vocab.stoi else unk_token_idx
                               for token in example.text]),
             dataset.fields['label'].vocab.stoi[example.label])
            for example in dataset.examples]

    return PytorchTextClassificationDataset(data, vocab)


def random_text_classification_dataset(num_samples, max_length=60, num_classes=2):

    # TODO: might need to provide a real vocab object here
    vocab = None
    vocab_size = 100

    data = []
    for i in range(num_samples):
        sample_length = np.random.randint(1, max_length)
        text = torch.cat([
            torch.randint(vocab_size, (sample_length,), dtype=torch.int) + 1,
            torch.tensor([0] * (max_length - sample_length), dtype=torch.int)
        ])
        label = np.random.randint(num_classes)

        data.append((text, label))

    return PytorchTextClassificationDataset(data, vocab)


def random_transformer_dataset(num_samples, max_length=60, num_classes=2, num_tokens=1000,
                               infer_labels=False):
    data = []
    for i in range(num_samples):
        sample_length = np.random.randint(1, max_length)
        text = torch.cat([
            torch.randint(num_tokens, (sample_length,), dtype=torch.int) + 1,
            torch.tensor([0] * (max_length - sample_length), dtype=torch.int)
        ])
        mask = torch.cat([
            torch.tensor([1] * sample_length, dtype=torch.int),
            torch.tensor([0] * (max_length - sample_length), dtype=torch.int)
        ])
        label = np.random.randint(num_classes)

        data.append((text, mask, label))

    y = [d[TransformersDataset.INDEX_LABEL] if d[TransformersDataset.INDEX_LABEL] is not None
         else TransformersDataset.NO_LABEL
         for d in data]
    target_labels = None if infer_labels else np.unique(y)

    return TransformersDataset(data, target_labels=target_labels)


def twenty_news_transformers(n, num_labels=10, subset='train'):
    train = fetch_20newsgroups(subset=subset)
    train_x = train.data[:n]
    train_y = np.random.randint(0, num_labels, size=n)

    tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-distilroberta-base')

    data = []
    for i, doc in enumerate(train_x):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            max_length=20,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )
        data.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], train_y[i]))

    return TransformersDataset(data)
