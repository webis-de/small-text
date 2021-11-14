import torch
import numpy as np

from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset

from examples.data.corpus_twenty_news import get_twenty_newsgroups_corpus


def get_train_test():
    train, test = get_twenty_newsgroups_corpus()

    try:
        from torchtext import data
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False, unk_token=None, pad_token=None)

    except AttributeError:
        # torchtext >= 0.8.0
        from torchtext.legacy import data
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False, unk_token=None)

    fields = [('text', text_field), ('label', label_field)]

    train = data.Dataset([data.Example.fromlist([text, labels], fields)
                          for text, labels in zip(train.data, train.target)],
                         fields)
    test = data.Dataset([data.Example.fromlist([text, labels], fields)
                         for text, labels in zip(test.data, test.target)],
                        fields)

    text_field.build_vocab(train, min_freq=1)
    label_field.build_vocab(train)

    train_tc = _dataset_to_text_classification_dataset(train)
    test_tc = _dataset_to_text_classification_dataset(test)

    return train_tc, test_tc


def _dataset_to_text_classification_dataset(dataset):
    assert dataset.fields['text'].vocab.itos[0] == '<unk>'
    assert dataset.fields['text'].vocab.itos[1] == '<pad>'
    unk_token_idx = 1

    vocab = dataset.fields['text'].vocab
    labels = list(set(dataset.fields['label'].vocab.itos))
    labels = np.array(labels)

    data = [(torch.LongTensor([vocab.stoi[token] if token in vocab.stoi else unk_token_idx
                               for token in example.text]),
             dataset.fields['label'].vocab.stoi[example.label])
            for example in dataset.examples]

    return PytorchTextClassificationDataset(data, vocab, target_labels=labels)
