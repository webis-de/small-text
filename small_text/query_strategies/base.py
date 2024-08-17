from __future__ import annotations
from abc import ABC, abstractmethod

import logging
import numpy as np
import numpy.typing as npt

from enum import Enum
from functools import partial, wraps
from typing import Union

from scipy.sparse import csr_matrix

from small_text.classifiers import Classifier
from small_text.data import Dataset
from small_text.query_strategies.exceptions import EmptyPoolException, PoolExhaustedException


class QueryStrategy(ABC):
    """Abstract base class for Query Strategies."""

    @abstractmethod
    def query(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix],
              n: int = 10) -> np.ndarray:
        """
        Queries instances from the unlabeled pool.

        A query selects instances from the unlabeled pool.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[uint]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[uint]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[uint] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.
        n : int
            Number of samples to query.

        Returns
        -------
        indices : numpy.ndarray
            Indices relative to `dataset` which were selected.
        """
        pass

    @staticmethod
    def _validate_query_input(indices_unlabeled: npt.NDArray[np.uint], n: int) -> None:

        if len(indices_unlabeled) == 0:
            raise EmptyPoolException('No unlabeled indices available. Cannot query an empty pool.')

        if n > len(indices_unlabeled):
            raise PoolExhaustedException('Pool exhausted: {} available / {} requested'
                                         .format(len(indices_unlabeled), n))


class ScoringMixin(ABC):
    """Provides scoring methods to a query strategy. In this context, "scoring" means that each instance
    in the dataset can be scored with numeric value."""

    @property
    @abstractmethod
    def last_scores(self) -> Union[npt.NDArray[np.double], None]:
        """Returns the scores that have been computed during the last `score()` call.

        Returns
        -------
        score : np.ndarray[float] or None
            Array of scores in the shape (n_samples, n_classes).
        """
        pass

    @abstractmethod
    def score(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix]) -> npt.NDArray[np.double]:
        """Assigns a numeric score to each instance in the given dataset.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[uint]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[uint]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[uint] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.

        Returns
        -------
        scores : np.ndarray[double]
            Array of scores in the shape (N,) where N is the size of `dataset` or a subset thereof.
        """


class ClassificationType(Enum):
    """Represents the classification type which can be either single-label or multi-label."""
    SINGLE_LABEL = 'single-label'
    MULTI_LABEL = 'multi-label'

    @staticmethod
    def from_str(classification_type_str: str) -> ClassificationType:
        """Creates a ClassificationType instance from the given string `classification_type_str`.

        Parameters
        ----------
        classification_type_str : {'single-label', 'multi-label'}
        """
        if classification_type_str == 'single-label':
            return ClassificationType.SINGLE_LABEL
        elif classification_type_str == 'multi-label':
            return ClassificationType.MULTI_LABEL
        else:
            raise ValueError('Cannot convert string to classification type enum: '
                             f'{classification_type_str}')


def constraints(cls=None, classification_type: Union[None, str, ClassificationType] = None):
    """Restricts a query strategy to certain settings such as single- or multi-label classification.

    This should be used sparingly and mostly in cases where a misconfiguration would not raise
    an error but is clearly unwanted.
    """
    if not callable(cls):
        return partial(constraints, classification_type=classification_type)

    @wraps(cls, updated=())
    class QueryStrategyConstraints(cls):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def query(self, clf, datasets, indices_unlabeled, indices_labeled, y, *args, n=10, **kwargs):

            if classification_type is not None:
                if isinstance(classification_type, str):
                    classification_type_ = ClassificationType.from_str(classification_type)
                else:
                    classification_type_ = classification_type

                if classification_type_ == ClassificationType.SINGLE_LABEL and isinstance(y, csr_matrix):
                    raise RuntimeError(f'Invalid configuration: This query strategy requires '
                                       f'classification_type={str(classification_type_.value)} '
                                       f'but multi-label data was encountered')
                elif classification_type_ == ClassificationType.MULTI_LABEL \
                        and not isinstance(y, csr_matrix):
                    raise RuntimeError(f'Invalid configuration: This query strategy requires '
                                       f'classification_type={str(classification_type_.value)} '
                                       f'but single-label data was encountered')

            return super().query(clf, datasets, indices_unlabeled, indices_labeled, y,
                                 *args, n=n, **kwargs)

    return QueryStrategyConstraints


def argselect(arr, n, maximum=True, tiebreak=True):
    """Returns the n highest or lowest indices in `arr`.

    This is intended to be a generalized `np.argpartition()` with tiebreaking functionality.

    Parameters
    ----------
    arr : np.ndarray[double]
        Array from which the indices are selected.
    n : int
        Number of indices to select.
    maximum : bool
        Select the n highest values if `maximum` is `True`, otherwise the n lowest values.
    tiebreak : bool
        Resolves tiebreaks if `True`, otherwise returns to np.arpartition / np.argsort.
    """
    if n == arr.shape[0]:
        return np.arange(n)
    elif n > arr.shape[0]:
        raise ValueError(f'n={n} out of bounds of array with shape {arr.shape}')

    if not tiebreak:
        return _argpartition(arr, n, maximum=maximum)[:n]

    return _argselect_tiebreak(arr, n, maximum=maximum)


def _argselect_tiebreak(arr, n, maximum=True):

    indices_argpartitioned_window = _argpartition(arr, n, maximum=maximum)
    if maximum:
        tiebreak_value = np.sort(-arr[indices_argpartitioned_window[:n]])[n-1]
    else:
        tiebreak_value = np.sort(arr[indices_argpartitioned_window[:n]])[n-1]

    k = 2 * n
    while k < arr.shape[0] and arr[indices_argpartitioned_window][k-1] == tiebreak_value:
        indices_argpartitioned_window = _argpartition(arr, k, maximum=maximum)
        k = min(k+n, arr.shape[0])

    indices_argpartitioned_window = indices_argpartitioned_window[:k]
    indices_tiebreak_value = np.isclose(arr[indices_argpartitioned_window], np.array([tiebreak_value] * k))
    if indices_tiebreak_value.sum() <= 1:
        # no tiebreak necessary
        return _argpartition(arr, n, maximum=maximum)[:n]

    # randomize the "tiebreak value's" occurrences
    indices_tiebreak_value_shuffled = np.copy(indices_argpartitioned_window[indices_tiebreak_value])
    np.random.shuffle(indices_tiebreak_value_shuffled)

    logging.debug(f'Tie breaking applied on {indices_tiebreak_value.sum()} equal scores.')
    indices_argpartitioned_window[indices_tiebreak_value] = indices_tiebreak_value_shuffled
    indices_argpartioned_n = _argpartition(arr[indices_argpartitioned_window], n, maximum=maximum)

    return indices_argpartitioned_window[indices_argpartioned_n[:n]]


def _argpartition(arr, k, maximum=True):
    if arr.shape[0] > k:
        if maximum is True:
            indices = np.argpartition(-arr, k)
        else:
            indices = np.argpartition(arr, k)
    else:
        if maximum is True:
            indices = np.argsort(-arr)
        else:
            indices = np.argsort(arr)
    return indices
