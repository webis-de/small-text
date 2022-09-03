import numpy as np

from collections import Counter
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups

from small_text.data.datasets import SklearnDataset
from small_text.data.sampling import _get_class_histogram

try:
    import torch
    from torchtext.vocab import Vocab

    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from small_text.integrations.transformers.datasets import TransformersDataset
except ImportError:
    pass

try:
    from transformers import AutoTokenizer
except ImportError:
    pass


def random_matrix_data(matrix_type, label_type, num_samples=100, num_dimension=40, num_labels=2):
    if matrix_type == 'dense':
        x = np.random.rand(num_samples, num_dimension)
    elif matrix_type == 'sparse':
        x = sparse.random(num_samples, num_dimension, density=0.15, format='csr')
    else:
        raise ValueError(f'Invalid matrix_type: {matrix_type}')

    if label_type == 'dense':
        y = np.random.randint(0, high=num_labels, size=x.shape[0])
    elif label_type == 'sparse':
        y = sparse.random(num_samples, num_labels, density=0.5, format='csr')
        y.data[np.s_[:]] = 1
        y = y.astype(int)
    else:
        raise ValueError(f'Invalid label_type: {label_type}')

    return x, y


# TODO: is this obsolete?
def random_labeling(num_classes, multi_label=False):
    label_values = np.arange(num_classes)
    if multi_label:
        num_labels = np.random.randint(num_classes)
        label = np.random.choice(label_values, num_labels, replace=False).tolist()
    else:
        label = np.random.randint(num_classes)
    return label


def random_labels(num_samples, num_classes, multi_label=False):
    if multi_label:
        y = sparse.random(num_samples, num_classes, density=0.5, format='csr')
        y.data[np.s_[:]] = 1
        y = y.astype(int)
    else:
        y = np.random.randint(0, high=num_classes, size=num_samples)
    return y


def random_sklearn_dataset(num_samples, vocab_size=60, num_classes=2, multi_label=False):

    x = sparse.random(num_samples, vocab_size, density=0.15, format='csr')

    if multi_label:
        y = sparse.random(num_samples, num_classes, density=0.5, format='csr')
        y.data[np.s_[:]] = 1
        y = y.astype(int)
    else:
        y = np.random.randint(0, high=num_classes, size=x.shape[0])

    return SklearnDataset(x, y)


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


def random_text_classification_dataset(num_samples=10, max_length=60, num_classes=2,
                                       multi_label=False, vocab_size=10,
                                       device='cpu', target_labels='inferred', dtype=torch.long):

    if target_labels not in ['explicit', 'inferred']:
        raise ValueError(f'Invalid test parameter value for target_labels: {str(target_labels)}')
    if num_classes > num_samples:
        raise ValueError('When num_classes is greater than num_samples the occurrence of each '
                         'class cannot be guaranteed')

    vocab = Vocab(Counter([f'word_{i}' for i in range(vocab_size)]))

    if multi_label:
        data = [(
                    torch.randint(vocab_size, (max_length,), dtype=dtype, device=device),
                    np.sort(random_labeling(num_classes, multi_label)).tolist()
                 )
                for _ in range(num_samples)]
    else:
        data = [
            (torch.randint(10, (max_length,), dtype=dtype, device=device),
             random_labeling(num_classes, multi_label))
            for _ in range(num_samples)]

    data = assure_all_labels_occur(data, num_classes, multi_label=multi_label)

    target_labels = None if target_labels == 'inferred' else np.arange(num_classes)
    return PytorchTextClassificationDataset(data, vocab, multi_label=multi_label,
                                            target_labels=target_labels)


def assure_all_labels_occur(data, num_classes, multi_label=False):
    """Enforces that all labels occur in the data."""
    label_list = [labels for *_, labels in data
                  if isinstance(labels, int) or len(labels) > 0]
    if len(label_list) == 0:
        return data

    if not np.all([isinstance(element, int) for element in label_list]):
        all_labels = np.concatenate(label_list, dtype=int)
        all_labels = all_labels.flatten()
    else:
        all_labels = np.array(label_list)

    hist = _get_class_histogram(all_labels, num_classes)
    missing_labels = np.arange(hist.shape[0])[hist == 0]

    for i, label_idx in enumerate(missing_labels):
        if multi_label:
            data[i] = data[i][:-1] + (np.sort(np.append(data[i][-1], [label_idx])),)
        else:
            data[i] = data[i][:-1] + (label_idx,)

    return data


def random_transformer_dataset(num_samples, max_length=60, num_classes=2, multi_label=False,
                               num_tokens=1000, target_labels='inferred', dtype=torch.long):

    if target_labels not in ['explicit', 'inferred']:
        raise ValueError(f'Invalid test parameter value for target_labels: {str(target_labels)}')

    data = []
    for i in range(num_samples):
        sample_length = np.random.randint(1, max_length)
        text = torch.cat([
            torch.randint(num_tokens, (sample_length,), dtype=dtype) + 1,
            torch.tensor([0] * (max_length - sample_length), dtype=dtype)
        ]).unsqueeze(0)
        mask = torch.cat([
            torch.tensor([1] * sample_length, dtype=dtype),
            torch.tensor([0] * (max_length - sample_length), dtype=dtype)
        ]).unsqueeze(0)
        if multi_label:
            labels = np.sort(random_labeling(num_classes, multi_label))
        else:
            labels = random_labeling(num_classes, multi_label)

        data.append((text, mask, labels))

    data = assure_all_labels_occur(data, num_classes, multi_label=multi_label)

    target_labels = None if target_labels == 'inferred' else np.arange(num_classes)
    return TransformersDataset(data, multi_label=multi_label, target_labels=target_labels)


def twenty_news_transformers(n, num_labels=10, subset='train', device='cpu'):
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
        encoded_dict = encoded_dict.to(device)
        data.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], train_y[i]))

    return TransformersDataset(data)
