import numpy as np
from small_text.base import LABEL_UNLABELED


def list_length(x):
    if hasattr(x, 'shape'):
        return x.shape[0]
    else:
        return len(x)


def check_training_data(train_set, validation_set):
    """Checks the given datasets' labels for validity.

    train_set : small_text.data.Dataset
        Any dataset.
    validation set : small_text.data.Dataset, optional
        Any dataset or None.
    """

    if train_set.is_multi_label:
        if np.array([(row.toarray() == LABEL_UNLABELED).any() for row in train_set.y]).any():
            raise ValueError('Training set labels must be labeled (greater or equal zero)')
        if validation_set is not None \
                and np.array([(row.toarray() == LABEL_UNLABELED).any()
                              for row in validation_set.y]).any():
            raise ValueError('Validation set labels must be labeled (greater or equal zero)')
    else:
        if (train_set.y == LABEL_UNLABELED).any():
            raise ValueError('Training set labels must be labeled (greater or equal zero)')
        if validation_set is not None and (validation_set.y == LABEL_UNLABELED).any():
            raise ValueError('Validation set labels must be labeled (greater or equal zero)')
