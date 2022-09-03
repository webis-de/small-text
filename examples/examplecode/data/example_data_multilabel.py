from sklearn.feature_extraction.text import TfidfVectorizer

from small_text.data import SklearnDataset
from small_text.utils.labels import list_to_csr

from examplecode.data.dataset_go_emotions import get_go_emotions_dataset


NUM_LABELS = 27 + 1


def get_train_test():
    return get_go_emotions_dataset()


def preprocess_data_sklearn(train, test):

    vectorizer = TfidfVectorizer(stop_words='english')

    y_train = list_to_csr(train['labels'], shape=(len(train), NUM_LABELS))
    y_test = list_to_csr(test['labels'], shape=(len(test), NUM_LABELS))

    ds_train = SklearnDataset.from_arrays(train['text'], y_train, vectorizer, train=True)
    ds_test = SklearnDataset.from_arrays(test['text'], y_test, vectorizer, train=False)

    return ds_train, ds_test
