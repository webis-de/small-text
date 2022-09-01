from sklearn.feature_extraction.text import TfidfVectorizer
from small_text.data import SklearnDataset

from examplecode.data.corpus_twenty_news import get_twenty_newsgroups_corpus


def get_train_test():
    return get_twenty_newsgroups_corpus()


def preprocess_data(train, test):
    vectorizer = TfidfVectorizer(stop_words='english')

    x_train = vectorizer.fit_transform(train.data)
    x_test = vectorizer.transform(test.data)

    return SklearnDataset(x_train, train.target), SklearnDataset(x_test, test.target)
