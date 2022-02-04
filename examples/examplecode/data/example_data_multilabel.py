from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from small_text.data import SklearnDataset
from small_text.utils.labels import list_to_csr

from examplecode.data.dataset_go_emotions import get_go_emotions_dataset


NUM_LABELS = 27 + 1


def get_train_test():
    return get_go_emotions_dataset()


def preprocess_data_sklearn(train, test):

    vectorizer = TfidfVectorizer(stop_words='english')

    x_train = normalize(vectorizer.fit_transform(train['text']))
    x_test = normalize(vectorizer.transform(test['text']))

    y_train = list_to_csr(train['labels'], shape=(len(train), NUM_LABELS))
    y_test = list_to_csr(test['labels'], shape=(len(test), NUM_LABELS))

    return SklearnDataset(x_train, y_train), SklearnDataset(x_test, y_test)
