from small_text import PytorchTextClassificationDataset

from examplecode.data.corpus_twenty_news import get_twenty_newsgroups_corpus


def get_train_test():
    return get_twenty_newsgroups_corpus(categories=['rec.sport.baseball', 'sci.med', 'rec.autos'])


def preprocess_data(train, test):

    try:
        from torchtext import data
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False, unk_token=None, pad_token=None)

    except AttributeError:
        # torchtext >= 0.8.0
        from torchtext.legacy import data
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False, unk_token=None)

    label_field.build_vocab(train)

    ds_train = PytorchTextClassificationDataset.from_arrays(train.data, train.target, text_field)
    ds_test = PytorchTextClassificationDataset.from_arrays(test.data, test.target, text_field,
                                                           train=False)

    return ds_train, ds_test
