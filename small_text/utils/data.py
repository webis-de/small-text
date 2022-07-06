import numpy as np
from small_text.base import LABEL_UNLABELED


def list_length(x):
    if hasattr(x, 'shape'):
        return x.shape[0]
    else:
        return len(x)


def check_training_data(train_set, validation_set, weights=None):
    """Checks the labels of the given datasets (and optionally the train weights) for validity.

    Validity in this context means that no `LABEL_UNLABELED` must not occur in the labels of both
    given datasets which usually should not happen anyway.

    If you encounter a ValueError this indicates that there is likely a bug in your code.

    train_set : small_text.data.Dataset
        Any dataset.
    validation set : small_text.data.Dataset, optional
        Any dataset or None.
    weights : np.ndarray[np.float32] or None, default=None
        Sample weights.

    Raises
    ------
    ValueError
        If either the given train set or validation set contains contains a `LABEL_UNLABELED` label.
    """

    if train_set.is_multi_label is False:
        if (train_set.y == LABEL_UNLABELED).any():
            raise ValueError('Training set labels must be labeled (greater or equal zero)')
        if validation_set is not None and (validation_set.y == LABEL_UNLABELED).any():
            raise ValueError('Validation set labels must be labeled (greater or equal zero)')

    if weights is not None:
        if len(train_set) != weights.shape[0]:
            raise ValueError('Training data and weights must have the same size.')

        if not np.all(weights > 0):
            raise ValueError('Weights must be greater zero.')
