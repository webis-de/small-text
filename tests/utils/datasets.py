import string
import numpy as np

from scipy import sparse
from sklearn.datasets import fetch_20newsgroups

from small_text.data.datasets import SklearnDataset, TextDataset
from small_text.data.sampling import _get_class_histogram
from small_text.utils.labels import csr_to_list, list_to_csr

try:
    import torch
    TORCH_DTYPE_LONG = torch.long

    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from small_text.integrations.transformers.datasets import TransformersDataset
except ImportError:
    TORCH_DTYPE_LONG = None

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


def random_text_data(num_samples=100):
    x = [' '.join([''.join(np.random.choice(list(string.ascii_lowercase), np.random.randint(3, 9)).tolist())
                   for _ in range(np.random.randint(5, 12))])
         for _ in range(num_samples)]
    return x


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

    return SklearnDataset(x, y, target_labels=np.arange(num_classes))


def trec_dataset(vocab_size=10_000):
    import datasets
    trec_dataset = datasets.load_dataset('trec')

    num_classes = 6
    target_labels = np.arange(num_classes)

    tokenizer = _train_tokenizer(trec_dataset['train']['text'], vocab_size)

    return _dataset_to_text_classification_dataset(trec_dataset['train'], tokenizer, target_labels), \
        _dataset_to_text_classification_dataset(trec_dataset['test'], tokenizer, target_labels), tokenizer


def _train_tokenizer(text, vocab_size, unk_token='[UNK]', pad_token='[PAD]'):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.WordLevel(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])

    trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=[unk_token, pad_token])
    tokenizer.train_from_iterator(text, trainer=trainer)

    return tokenizer


def _dataset_to_text_classification_dataset(dataset, tokenizer, target_labels):
    import torch

    data = [(torch.LongTensor(tokenizer.encode(example).ids), label)
            for example, label in zip(dataset['text'], dataset['coarse_label'])]

    return PytorchTextClassificationDataset(data, target_labels=target_labels)


def random_text_classification_dataset(num_samples=10, max_length=60, num_classes=2, multi_label=False, vocab_size=10,
                                       device='cpu', target_labels='inferred', dtype=TORCH_DTYPE_LONG):

    if target_labels not in ['explicit', 'inferred']:
        raise ValueError(f'Invalid test parameter value for target_labels: {str(target_labels)}')

    if num_classes > num_samples:
        raise ValueError('When num_classes is greater than num_samples the occurrence of each '
                         'class cannot be guaranteed')

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
    return PytorchTextClassificationDataset(data, multi_label=multi_label, target_labels=target_labels)


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
            data[i] = data[i][:-1] + (np.sort(np.append(data[i][-1], [label_idx])).astype(int).tolist(),)
        else:
            data[i] = data[i][:-1] + (label_idx,)

    return data


def assure_all_labels_occur_numpy(y, num_classes, multi_label=False):
    """Enforces that all labels occur in the data."""

    if multi_label:
        y = csr_to_list(y)

    label_list = [labels for labels in y
                  if isinstance(labels, int) or (isinstance(labels, list) and len(labels) > 0)]
    if len(label_list) == 0:
        return y

    if not np.all([isinstance(element, int) for element in label_list]):
        all_labels = np.concatenate(label_list, dtype=int)
        all_labels = all_labels.flatten()
    else:
        all_labels = np.array(label_list)

    hist = _get_class_histogram(all_labels, num_classes)
    missing_labels = np.arange(hist.shape[0])[hist == 0]

    assert len(y) >= missing_labels.shape[0]
    for i, label_idx in enumerate(missing_labels):
        if multi_label:
            y[i] = np.sort(np.append(y[i], [label_idx])).tolist()
        else:
            y[i] = y[i][:-1] + (label_idx,)

    if multi_label:
        y = list_to_csr(y, (len(y), num_classes))

    return y


def random_transformer_dataset(num_samples, max_length=60, num_classes=2, multi_label=False,
                               num_tokens=1000, target_labels='inferred', dtype=TORCH_DTYPE_LONG):

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
            labels = np.sort(random_labeling(num_classes, multi_label)).tolist()
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


def twenty_news_text(n, num_classes=10, subset='train', multi_label=False):
    train = fetch_20newsgroups(subset=subset)
    x = train.data[:n]
    if multi_label:
        y = list_to_csr([np.sort(random_labeling(num_classes, multi_label)).tolist()
                         for _ in range(n)], (n, num_classes))
    else:
        y = np.random.randint(0, num_classes, size=n)
    y = assure_all_labels_occur_numpy(y, num_classes, multi_label=multi_label)
    return TextDataset(x, y)


def random_text_dataset(n, num_classes=10, multi_label=False, assure_all_labels_occur=True):
    x = random_text_data(n)
    if multi_label:
        y = list_to_csr([np.sort(random_labeling(num_classes, multi_label)).tolist()
                         for _ in range(n)], (n, num_classes))
    else:
        y = np.random.randint(0, num_classes, size=n)

    if assure_all_labels_occur:
        y = assure_all_labels_occur_numpy(y, num_classes, multi_label=multi_label)
    return TextDataset(x, y)


def set_y(dataset, indices_initial, y_initial):
    y_tmp = dataset.y
    if dataset.is_multi_label:
        y = csr_to_list(dataset.y)
        y_target= csr_to_list(y_initial)
        for i in np.arange(y_initial.shape[0]):
            y[indices_initial[i]] = y_target[i]
        y_tmp = list_to_csr(y, y_tmp.shape)
    else:
        y_tmp[indices_initial] = y_initial
    dataset.y = y_tmp

    return dataset
