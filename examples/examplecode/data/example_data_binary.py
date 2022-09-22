from sklearn.feature_extraction.text import TfidfVectorizer
from small_text import SklearnDataset

from examplecode.data.corpus_twenty_news import get_twenty_newsgroups_corpus


def get_train_test():
    return get_twenty_newsgroups_corpus()


def preprocess_data(train, test):
    vectorizer = TfidfVectorizer(stop_words='english')

    ds_train = SklearnDataset.from_arrays(train.data, train.target, vectorizer, train=True)
    ds_test = SklearnDataset.from_arrays(test.data, test.target, vectorizer, train=False)

    return ds_train, ds_test
