import warnings
import numpy as np

from abc import ABCMeta, abstractmethod
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from small_text.base import LABEL_UNLABELED
from small_text.data.exceptions import UnsupportedOperationException
from small_text.data.sampling import stratified_sampling, balanced_sampling
from small_text.utils.annotations import experimental
from small_text.utils.labels import get_flattened_unique_labels


def check_size(expected_num_samples, num_samples):
    if num_samples != expected_num_samples:
        raise ValueError(f'Size mismatch: expected {expected_num_samples} samples, '
                         f'encountered {num_samples} samples')


def check_dataset_and_labels(x, y):
    if x.shape[0] != y.shape[0]:
        raise ValueError(f'Feature and label dimensions do not match: '
                         f'x.shape = {x.shape}, y.shape= {y.shape} ### {type(x)} / {type(y)}')


def check_target_labels(target_labels):
    if target_labels is None:
        warnings.warn('Passing target_labels=None is discouraged as it can lead to '
                      'unintended results in combination with indexing and cloning. '
                      'Moreover, explicit target labels might be required in the '
                      'next major version.')


def get_updated_target_labels(is_multi_label, y, target_labels):

    if is_multi_label:
        new_labels = np.setdiff1d(np.unique(y.indices), target_labels)
    else:
        new_labels = np.setdiff1d(np.unique(y), target_labels)

    if new_labels.shape[0] > 0:
        target_labels = np.unique(np.union1d(target_labels, new_labels))

    return target_labels


class Dataset(metaclass=ABCMeta):
    """A dataset contains a set of instances in the form of features, include a respective
    labeling for every instance."""

    @property
    @abstractmethod
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
    @abstractmethod
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
        """Assigns new labels to the existing instances.

        Note
        ----
        This can alter `self.target_labels` but the set of target labels only grows, it is never
        automatically reduced. This means, if labels in `y_new` occur, which are not in
        `self.target_labels`, they will be added during this operation.
        """
        pass

    @property
    @abstractmethod
    def is_multi_label(self):
        """Returns `True` if this is a multi-label dataset, otherwise `False`."""
        pass

    @property
    @abstractmethod
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

    @abstractmethod
    def clone(self):
        """Returns an identical copy of the dataset.

        Returns
        -------
        dataset : Dataset
            An exact copy of the dataset.
        """
        pass


class DatasetView(metaclass=Dataset.__class__):
    """An immutable view on a Dataset or a subset thereof."""

    @property
    @abstractmethod
    def dataset(self):
        pass


class SklearnDatasetView(DatasetView):
    """An immutable view on a SklearnDataset or a subset thereof."""

    def __init__(self, dataset, selection):
        """
        Parameters
        ----------
        dataset : Dataset
            The base dataset.
        selection : int or list or slice or np.ndarray
            Selects the subset for this view.
        """
        self.obj_class = type(self)
        self._dataset = dataset

        self.selection = selection

    @property
    def dataset(self):
        return self._dataset

    @property
    def x(self):
        return self.dataset.x[select(self.dataset, self.selection)]

    @x.setter
    def x(self, x):
        raise UnsupportedOperationException('Cannot set x on a DatasetView')

    @property
    def y(self):
        y_result = self.dataset.y[self.selection]
        if self.is_multi_label:
            return y_result
        else:
            return np.atleast_1d(y_result)

    @y.setter
    def y(self, y):
        raise UnsupportedOperationException('Cannot set y on a DatasetView')

    @property
    def is_multi_label(self):
        return self.dataset.is_multi_label

    @property
    def target_labels(self):
        return self.dataset.target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        raise UnsupportedOperationException('Cannot set target_labels on a DatasetView')

    def clone(self):
        if isinstance(self.x, csr_matrix):
            x = self.x.copy()
        else:
            x = np.copy(self.x)

        if isinstance(self.y, csr_matrix):
            y = self.y.copy()
        else:
            y = np.copy(self.y)

        dataset = self
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        if dataset.track_target_labels:
            target_labels = None
        else:
            target_labels = np.copy(dataset._target_labels)

        return SklearnDataset(x,
                              y,
                              target_labels=target_labels)

    def __getitem__(self, item):
        return self.obj_class(self, item)

    def __len__(self):
        return self.dataset.x[select(self.dataset, self.selection)].shape[0]


def is_multi_label(y):
    if isinstance(y, csr_matrix):
        return True
    else:
        return False


def select(dataset, selection):
    if isinstance(selection, (int, np.integer)) and not issparse(dataset.x):
        return np.s_[np.newaxis, selection]
    return selection


class SklearnDataset(Dataset):
    """A dataset representations which is usable in combination with scikit-learn classifiers.
    """

    def __init__(self, x, y, target_labels=None):
        """
        Parameters
        ----------
        x : numpy.ndarray or scipy.sparse.csr_matrix
            Dense or sparse feature matrix.
        y : numpy.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be inferred from `y` if `None` is passed."""
        check_dataset_and_labels(x, y)
        check_target_labels(target_labels)

        self._x = x
        self._y = y

        self.multi_label = is_multi_label(self._y)

        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = target_labels
        else:
            self.track_target_labels = True
            self._infer_target_labels()

    def _infer_target_labels(self):
        if self.x.shape[0] == 0:
            self.target_labels = np.array([0])
        else:
            unique_labels = np.unique(self._get_flattened_labels())
            if unique_labels.shape[0] > 0:
                max_label_id = unique_labels.max()
                self.target_labels = np.arange(max_label_id+1)
            else:
                self.target_labels = np.array([0])

    def _get_flattened_labels(self):
        if self.multi_label:
            return self.y.indices
        else:
            return self.y[np.argwhere(self.y > LABEL_UNLABELED)]

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
        check_dataset_and_labels(x, self._y)
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        check_dataset_and_labels(self.x, y)
        self._y = y

        if self.track_target_labels:
            self.target_labels = get_updated_target_labels(self.is_multi_label, y, self.target_labels)
        else:
            max_label_id = np.max(y)
            max_target_labels_id = self.target_labels.max()
            if max_label_id > max_target_labels_id:
                raise ValueError(f'Error while assigning new labels to dataset: '
                                 f'Encountered label with id {max_label_id} which is outside of '
                                 f'the configured set of target labels (whose maximum label is '
                                 f'is {max_target_labels_id}) [track_target_labels=False]')

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
        encountered_labels = get_flattened_unique_labels(self)
        if np.setdiff1d(encountered_labels, target_labels).shape[0] > 0:
            raise ValueError('Cannot remove existing labels from target_labels as long as they '
                             'still exists in the data. Create a new dataset instead.')
        self._target_labels = target_labels

    def clone(self):
        if isinstance(self._x, csr_matrix):
            x = self._x.copy()
        else:
            x = np.copy(self._x)

        if isinstance(self._y, csr_matrix):
            y = self._y.copy()
        else:
            y = np.copy(self._y)

        if self.track_target_labels:
            target_labels = None
        else:
            target_labels = np.copy(self._target_labels)

        return SklearnDataset(x, y, target_labels=target_labels)

    @classmethod
    @experimental
    def from_arrays(cls, texts, y, vectorizer, target_labels=None, train=True):
        """Constructs a new SklearnDataset from the given text and label arrays.

        Parameters
        ----------
        texts : list of str or np.ndarray[str]
            List of text documents.
        y : np.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
            Depending on the type of `y` the resulting dataset will be single-label (`np.ndarray`)
            or multi-label (`scipy.sparse.csr_matrix`).
        vectorizer : object
            A scikit-learn vectorizer which is used to construct the feature matrix.
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be directly passed to the datset constructor.
        train : bool
            If `True` fits the vectorizer and transforms the data, otherwise just transforms the
            data.

        Returns
        -------
        dataset : SklearnDataset
            A dataset constructed from the given texts and labels.


        .. seealso::
           `scikit-learn docs: Vectorizer API reference
           <https://scikit-learn.org/stable/modules/classes.html
           #module-sklearn.feature_extraction.text>`_

        .. warning::
           This functionality is still experimental and may be subject to change.

        .. versionadded:: 1.1.0
        """
        if train:
            x = vectorizer.fit_transform(texts)
        else:
            x = vectorizer.transform(texts)

        return SklearnDataset(x, y, target_labels=target_labels)

    def __getitem__(self, item):
        return SklearnDatasetView(self, item)

    def __len__(self):
        return self._x.shape[0]


def split_data(train_set, y=None, strategy='random', validation_set_size=0.1, return_indices=False):
    """Splits the given set `train_set` into two subsets (`sub_train` and `sub_valid`) according to
    a specified strategy.

    Parameters
    ----------
    train_set : Dataset
        A training dataset that should be split.
    y : np.ndarray[int]
        Labels for the train set.
    strategy : {'random', 'balanced', 'stratified'}, default='random'
        The strategy used for splitting.
    validation_set_size : float, default=0.1
        Fraction of the input size that will be the size of the validation set.
    return_indices : bool, default=False
        Returns two lists (np.ndarray[int]) of indices instead of two subsets if True.

    Returns
    -------
    train_split_or_indices : Dataset or numpy.ndarray[int]
        The train split or indices (relative to `train_set`) defining the train split.
    validation_split_or_indices : Dataset or numpy.ndarray[int]
        The validation split or indices (relative to `train_set`) defining the validation split.

    Note
    ----
    Labes are passed separately due to legacy reasons. This circumstance is currently also used to
    handle the multi-label case. This might change in the future.
    """
    if validation_set_size == 0 or validation_set_size >= 1.0:
        raise ValueError('Invalid value encountered for "validation_set_size". '
                         'Must be within the interval (0.0, 1.0).')

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
