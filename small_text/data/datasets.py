from __future__ import annotations

import warnings
import numpy as np


from abc import ABCMeta, abstractmethod
from copy import copy

from typing import Generic, Union, Sized

from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from small_text.base import LABEL_UNLABELED
from small_text.data.exceptions import UnsupportedOperationException
from small_text.data._typing import (
    DATA,
    LABELS,
    VIEW,
    SKLEARN_DATA
)
from small_text.utils.data import list_length


def check_size(expected_num_samples, num_samples):
    if num_samples != expected_num_samples:
        raise ValueError(f'Size mismatch: expected {expected_num_samples} samples, '
                         f'encountered {num_samples} samples')


def check_dataset_and_labels(x, y):
    len_x = list_length(x)
    if len_x != y.shape[0]:
        if hasattr(x, 'shape'):
            raise ValueError(f'Feature and label dimensions do not match: '
                             f'x.shape = {x.shape}, y.shape= {y.shape} ### {type(x)} / {type(y)}')
        else:
            raise ValueError(f'Feature and label dimensions do not match: '
                             f'x = ({len_x},), y.shape= ({y.shape[0]},) ### {type(x)} / {type(y)}')


def check_text_data(x):
    for i, item in enumerate(x):
        if item is None:
            raise ValueError(f'instance #{i} is None which is not allowed.')


def check_target_labels(target_labels):
    if target_labels is None:
        warnings.warn('Passing target_labels=None is discouraged as it can lead to '
                      'unintended results in combination with indexing and cloning. '
                      'Moreover, explicit target labels might be required in the '
                      'next major version.',
                      stacklevel=2)


def get_updated_target_labels(is_multi_label, y, target_labels):

    if is_multi_label:
        new_labels = np.setdiff1d(np.unique(y.indices), target_labels)
    else:
        new_labels = np.setdiff1d(np.unique(y), target_labels)

    if new_labels.shape[0] > 0:
        target_labels = np.unique(np.union1d(target_labels, new_labels))

    return target_labels


def _get_flattened_labels(y, multi_label=False):
    if multi_label:
        return y.indices
    else:
        return y[np.argwhere(y > LABEL_UNLABELED)]


def _infer_target_labels(x, y, multi_label=False):
    if list_length(x) == 0:
        return np.array([0])
    else:
        unique_labels = np.unique(_get_flattened_labels(y, multi_label=multi_label))
        if unique_labels.shape[0] > 0:
            max_label_id = unique_labels.max()
            return np.arange(max_label_id+1)
        else:
            return np.array([0])


class Dataset(Sized, Generic[VIEW, DATA, LABELS], metaclass=ABCMeta):
    """A dataset contains a set of instances in the form of features, include a respective
    labeling for every instance."""

    @property
    @abstractmethod
    def x(self) -> DATA:
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
    def y(self) -> LABELS:
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
    def is_multi_label(self) -> bool:
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
    def clone(self) -> Dataset:
        """Returns an identical copy of the dataset.

        Returns
        -------
        dataset : Dataset
            An exact copy of the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, item) -> VIEW:
        pass


class DatasetView(Dataset[VIEW, DATA, LABELS], metaclass=Dataset.__class__):
    """An immutable view on a Dataset or a subset thereof."""

    @property
    @abstractmethod
    def dataset(self) -> Union[Dataset, DatasetView]:
        pass

    @property
    @abstractmethod
    def x(self):
        """Returns the features.

        Returns
        -------
        x : object
            Feature representation.
        """
        return super().x

    @x.setter
    def x(self, x_new):
        super().x = x_new

    @property
    @abstractmethod
    def y(self):
        """Returns the labels.

        Returns
        -------
        y : numpy.ndarray or scipy.sparse.csr_matrix
            The labels as either numpy array (single-label) or sparse matrix (multi-label).
        """
        return super().y

    @y.setter
    def y(self, y_new):
        """Assigns new labels to the existing instances.

        Note
        ----
        This can alter `self.target_labels` but the set of target labels only grows, it is never
        automatically reduced. This means, if labels in `y_new` occur, which are not in
        `self.target_labels`, they will be added during this operation.
        """
        super().y = y_new


class SklearnDatasetView(DatasetView[VIEW, SKLEARN_DATA, SKLEARN_DATA]):
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
    def dataset(self) -> Union[SklearnDataset, SklearnDatasetView]:
        return self._dataset

    @property
    def x(self):
        return self.dataset.x[select(self.dataset, self.selection)]

    @x.setter
    def x(self, x_new):
        raise UnsupportedOperationException('Cannot set x on a DatasetView')

    @property
    def y(self):
        y_result = self.dataset.y[self.selection]
        if self.is_multi_label:
            return y_result
        else:
            return np.atleast_1d(y_result)

    @y.setter
    def y(self, y_new):
        raise UnsupportedOperationException('Cannot set y on a DatasetView')

    @property
    def is_multi_label(self) -> bool:
        return self.dataset.is_multi_label

    @property
    def target_labels(self):
        return self.dataset.target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        raise UnsupportedOperationException('Cannot set target_labels on a DatasetView')

    def clone(self) -> SklearnDataset:
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
        return self.x.shape[0]


class TextDatasetView(DatasetView):
    """An immutable view on a TextDataset or a subset thereof."""

    def __init__(self, dataset: Dataset, selection: Union[int, slice, np.s_.__class__]):
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
        if isinstance(self.selection, (int, np.integer)):
            return [self.dataset.x[self.selection]]

        is_slice_or_list = isinstance(self.selection, (np.s_.__class__, slice, list))

        if is_slice_or_list:
            dataset_len = len(self.dataset)
            selection = np.arange(dataset_len)[self.selection]
        else:
            selection = self.selection

        return [self.dataset.x[i] for i in selection]

    @x.setter
    def x(self, x_new):
        raise UnsupportedOperationException('Cannot set x on a DatasetView')

    @property
    def y(self):
        y_result = self.dataset.y[self.selection]
        if self.is_multi_label:
            return y_result
        else:
            return np.atleast_1d(y_result)

    @y.setter
    def y(self, y_new):
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

        return TextDataset(x,
                           y,
                           target_labels=target_labels)

    def __getitem__(self, item):
        return self.obj_class(self, item)

    def __len__(self):
        return len(self.x)


def is_multi_label(y):
    if isinstance(y, csr_matrix):
        return True
    else:
        return False


def select(dataset, selection):
    if isinstance(selection, (int, np.integer)) and not issparse(dataset.x):
        return np.s_[np.newaxis, selection]
    return selection


class SklearnDataset(Dataset[VIEW, SKLEARN_DATA, SKLEARN_DATA]):
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
            self.target_labels = _infer_target_labels(self._x, self._y, multi_label=self.multi_label)

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
    def x(self, x_new):
        check_dataset_and_labels(x_new, self._y)
        self._x = x_new

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y_new):
        check_dataset_and_labels(self.x, y_new)
        self._y = y_new

        if self.track_target_labels:
            self.target_labels = get_updated_target_labels(self.is_multi_label, y_new, self.target_labels)
        else:
            max_label_id = np.max(y_new)
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
        labels = _get_flattened_labels(self._y, multi_label=self.multi_label)
        encountered_labels = np.setdiff1d(labels, np.array([LABEL_UNLABELED]))
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
           <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
           #sklearn.feature_extraction.text.TfidfVectorizer>`__

        .. versionadded:: 1.1.0
        """
        if train:
            x = vectorizer.fit_transform(texts)
        else:
            x = vectorizer.transform(texts)

        return SklearnDataset(x, y, target_labels=target_labels)

    def __getitem__(self, item) -> SklearnDatasetView:
        return SklearnDatasetView(self, item)

    def __len__(self):
        return self._x.shape[0]


class TextDataset(Dataset):
    """A dataset representation consisting of raw text data.
    """

    def __init__(self, x, y, target_labels=None):
        """
        Parameters
        ----------
        x : list of str
            List of texts.
        y : numpy.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be inferred from `y` if `None` is passed."""
        check_dataset_and_labels(x, y)
        check_text_data(x)
        check_target_labels(target_labels)

        if isinstance(x, np.ndarray):
            x = x.tolist()

        self._x = x
        self._y = y

        self.multi_label = is_multi_label(self._y)

        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = target_labels
        else:
            self.track_target_labels = True
            self.target_labels = _infer_target_labels(self._x, self._y, multi_label=self.multi_label)

    @property
    def x(self):
        """Returns the features.

        Returns
        -------
        x : list of str
            List of texts.
        """
        return self._x

    @x.setter
    def x(self, x_new):
        check_text_data(x_new)
        check_dataset_and_labels(x_new, self._y)
        self._x = x_new

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y_new):
        check_dataset_and_labels(self.x, y_new)
        self._y = y_new

        if self.track_target_labels:
            self.target_labels = get_updated_target_labels(self.is_multi_label, y_new, self.target_labels)
        else:
            max_label_id = np.max(y_new)
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
        labels = _get_flattened_labels(self._y, multi_label=self.multi_label)
        encountered_labels = np.setdiff1d(labels, np.array([LABEL_UNLABELED]))
        if np.setdiff1d(encountered_labels, target_labels).shape[0] > 0:
            raise ValueError('Cannot remove existing labels from target_labels as long as they '
                             'still exists in the data. Create a new dataset instead.')
        self._target_labels = target_labels

    def clone(self) -> TextDataset:
        x = copy(self._x)

        if isinstance(self._y, csr_matrix):
            y = self._y.copy()
        else:
            y = np.copy(self._y)

        if self.track_target_labels:
            target_labels = None
        else:
            target_labels = np.copy(self._target_labels)

        return TextDataset(x, y, target_labels=target_labels)

    @classmethod
    def from_arrays(cls, texts, y, target_labels=None):
        """Constructs a new TextDataset from the given text and label arrays.

        Parameters
        ----------
        texts : list of str
            List of text documents.
        y : np.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
            Depending on the type of `y` the resulting dataset will be single-label (`np.ndarray`)
            or multi-label (`scipy.sparse.csr_matrix`).
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be directly passed to the datset constructor.

        Returns
        -------
        dataset : SklearnDataset
            A dataset constructed from the given texts and labels.


        .. seealso::
           `scikit-learn docs: Vectorizer API reference
           <https://scikit-learn.org/stable/api/sklearn.feature_extraction.html
           #module-sklearn.feature_extraction.text>`_

        .. versionadded:: 1.2.0
        """

        return TextDataset(texts, y, target_labels=target_labels)

    def __getitem__(self, item):
        return TextDatasetView(self, item)

    def __len__(self):
        return len(self._x)
