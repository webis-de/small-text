import numpy as np

from abc import ABC
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from small_text.data.exceptions import UnsupportedOperationException
from small_text.data.sampling import stratified_sampling, balanced_sampling


def check_size(expected_num_samples, num_samples):
    if num_samples != expected_num_samples:
        raise ValueError(f'Size mismatch: expected {expected_num_samples} samples, '
                         f'encountered {num_samples} samples')


class Dataset(ABC):
    """Abstract class for all datasets."""

    @property
    def x(self):
        """Returns the features.

        Returns
        -------
        x : object
            Feature representation.
        """
        pass

    @x.setter
    def x(self, x_new):
        pass

    @property
    def y(self):
        """Returns the labels.

        Returns
        -------
        y : numpy.ndarray or scipy.sparse.csr_matrix
            The labels as either numpy array (single-label) or sparse matrix (multi-label).
        """
        pass

    @y.setter
    def y(self, y_new):
        pass

    @property
    def is_multi_label(self):
        """Returns `True` if this is a multi-label dataset, otherwise `False`."""
        pass

    @property
    def target_labels(self):
        """Returns a list of possible labels.

        Returns
        -------
        target_labels : numpy.ndarray
            List of possible labels.
        """
        pass

    @target_labels.setter
    def target_labels(self, target_labels):
        pass


class DatasetView(Dataset):
    """An immutable view on a Dataset or a subset thereof.

    Parameters
    ----------
    dataset : Dataset
        The base dataset.
    selection : int or list or slice or np.ndarray
        Selects the subset for this view.
    """
    def __init__(self, dataset, selection):
        self.obj_class = type(self)
        self._dataset = dataset

        if isinstance(selection, int):
            if issparse(dataset.x):
                self.selection = selection
            else:
                self.selection = np.s_[np.newaxis, selection]
        else:
            self.selection = selection

    @property
    def x(self):
        """Returns the features.

        Returns
        -------
        x : numpy.ndarray or scipy.sparse.csr_matrix
            Dense or sparse feature matrix.
        """
        return self._dataset.x[self.selection]

    @x.setter
    def x(self, x):
        raise UnsupportedOperationException('Cannot set x on a DatasetView')

    @property
    def y(self):
        return self._dataset.y[self.selection]

    @y.setter
    def y(self, y):
        raise UnsupportedOperationException('Cannot set y on a DatasetView')

    @property
    def is_multi_label(self):
        return self._dataset.is_multi_label

    @property
    def target_labels(self):
        """Returns a list of possible labels.

        Returns
        -------
        target_labels : numpy.ndarray
            List of possible labels.
        """
        return self._dataset.target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        raise UnsupportedOperationException('Cannot set target_labels on a DatasetView')

    def __getitem__(self, item):
        return self.obj_class(self, item)

    def __len__(self):
        return self._dataset.x[self.selection].shape[0]


def is_multi_label(y):
    if isinstance(y, csr_matrix):
        return True
    else:
        return False


class SklearnDataset(Dataset):
    """A dataset representations which is usable in combination with scikit-learn classifiers.

    Parameters
    ----------
    x : numpy.ndarray or scipy.sparse.csr_matrix
        Dense or sparse feature matrix.
    y : list of int
        List of labels where each label belongs to the features of the respective row.
    target_labels : list of int or None
        List of possible labels. Will be inferred from `y` if `None` is passed.
    """

    def __init__(self, x, y, target_labels=None):

        self._x = x
        if isinstance(y, list):
            self._y = np.array(y)
        else:
            self._y = y

        self.multi_label = is_multi_label(self._y)

        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = np.array(target_labels)
        else:
            self.track_target_labels = True
            self._infer_target_labels(self._y)

    def _infer_target_labels(self, y):
        if isinstance(y, csr_matrix):
            self.target_labels = np.unique(y.indices)
        else:
            self.target_labels = np.unique(y)

    @property
    def x(self):
        """Returns the features.

        Returns
        -------
        x : numpy.ndarray or scipy.sparse.csr_matrix
            Dense or sparse feature matrix.
        """
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
        if self.track_target_labels:
            self._infer_target_labels(self._y)

    @property
    def is_multi_label(self):
        return self.multi_label

    @property
    def target_labels(self):
        """Returns a list of possible labels.

        Returns
        -------
        target_labels : numpy.ndarray
            List of possible labels.
        """
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        # TODO: how to handle existing labels that are outside this set
        self._target_labels = target_labels

    def __getitem__(self, item):
        return DatasetView(self, item)

    def __len__(self):
        return self._x.shape[0]


def split_data(train_set, y=None, strategy='random', validation_set_size=0.1, return_indices=False):
    """
    Splits the given set `train_set` into two subsets (`sub_train` and `sub_valid`).

    Parameters
    ----------
    train_set :

    y : np.ndarray [int]
        Labels for the train set.
    strategy : str
        The strategy used for splitting. One of 'random', 'balanced', 'stratified'.
    validation_set_size : float
        Fraction of the input size that will be the size of the validation set.
    return_indices : bool
        Returns two lists (np.ndarray[int]) of indices instead of two subsets if True.
    """

    train_len = int(len(train_set) * (1-validation_set_size))

    if strategy == 'random':
        indices = np.random.permutation(len(train_set))
        indices_train = indices[:train_len]
        indices_valid = indices[train_len:]
    elif strategy == 'balanced':
        indices_valid = balanced_sampling(y, n_samples=len(train_set)-train_len)
        indices_train = list(range(len(train_set)))
        indices_train = np.array([i for i in indices_train if i not in set(indices_valid)])
    elif strategy == 'stratified':
        indices_valid = stratified_sampling(y, n_samples=len(train_set)-train_len)
        indices_train = list(range(len(train_set)))
        indices_train = np.array([i for i in indices_train if i not in set(indices_valid)])
    else:
        raise ValueError('Invalid strategy: ' + strategy)

    if return_indices:
        return indices_train, indices_valid
    else:
        return train_set[indices_train], train_set[indices_valid]
