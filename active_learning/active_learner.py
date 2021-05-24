from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from active_learning.exceptions import LearnerNotInitializedException
from active_learning.utils.data import list_length
from active_learning.version import __version__ as version


class ActiveLearner(ABC):
    """Abstract base class for Active Learners."""

    @abstractmethod
    def query(self, num_samples=10):
        pass

    @abstractmethod
    def update(self, y):
        pass


class AbstractPoolBasedActiveLearner(ActiveLearner):

    def query(self, num_samples=10):
        pass

    def update(self, y):
        pass

    @abstractmethod
    def initialize_data(self, x_indices_initial, y_initial, *args, **kwargs):
        """
        Initializes the pool and existing labelings.

        This methods needs to be called whenever the underlying data changes, in particularly before the first loop.

        Parameters
        ----------
        x_indices_initial : list of int
            Positional indices pointing at training examples. This is the intially labelled set
            for training an initial classifier.
        y_initial : list of int
            The respective labels belonging to the examples referenced by `x_indices_initial`.
        """
        pass

    @property
    @abstractmethod
    def classifier(self):
        pass

    @property
    @abstractmethod
    def query_strategy(self):
        pass


class PoolBasedActiveLearner(AbstractPoolBasedActiveLearner):
    """
    A pool-based active learner in which a pool holds all available unlabeled data.
    It uses a classifier, a query strategy and manages the mutually exclusive partition over the
    whole training data into labeled and unlabeled.

    Parameters
    ----------
    clf_factory : active_learning.classifiers.factories.AbstractClassifierFactory
        A factory responsible for creating new classifier instances.
    query_strategy : active_learning.query_strategies.QueryStrategy
        Query strategy which is responsible for selecting instances during a `query()` call.
    x_train : active_learning.data.Dataset
        A training dataset that is supported by the underlying classifier.
    incremental_training : bool
        If False, creates and trains a new classifier only before the first query,
        otherwise re-trains the existing classifier. Incremental training must be supported
        by the classifier provided by `clf_factory`.

    Attributes
    ----------
    x_indices_labeled : numpy.ndarray
        Indices of instances constituting the labeled pool (relative to `self.x_train`).
    y : numpy.ndarray
        Labels for the the current labeled pool. Each tuple `(x_indices_labeled[i], y[i])`
        represents one labeled sample.
    queried_indices : numpy.ndarray or None
        Queried indices returned by the last `query()` call, or `None` if no query has been
        executed yet.
    """
    def __init__(self, clf_factory, query_strategy, x_train, incremental_training=False):
        self._clf = None
        self._clf_factory = clf_factory
        self._query_strategy = query_strategy

        self._label_to_position = None

        self.x_train = x_train
        self.incremental_training = incremental_training

        self.x_indices_labeled = np.empty(shape=(0), dtype=int)
        self.y = np.empty(shape=(0), dtype=int)
        self.queried_indices = None

    def initialize_data(self, x_indices_initial, y_initial, x_indices_validation=None,
                        retrain=True):
        """
        (Re-)Initializes the current labeled pool.

        This is required once before the first `query()` call, and whenever the labeled pool
        is changed from the outside, i.e. when `self.x_train` changes.

        Parameters
        ----------
        x_indices_initial : numpy.ndarray
            A list of indices (relative to `self.x_train`) of initially labeled samples.
        y_initial : numpy.ndarray
            List of labels. One label correspongs to each index in `x_indices_initial`.
        x_indices_validation : numpy.ndarray
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. Otherwise each classifier that uses a validation set will be responsible
            for creating a validation set. Only used if `retrain=True`.
        retrain : bool
            Retrains the model after the update if True.
        """
        self.x_indices_labeled = x_indices_initial
        self._label_to_position = self._build_label_to_position_index()
        self.y = y_initial

        if retrain:
            self._retrain(x_indices_validation=x_indices_validation)

    def query(self, num_samples=10, x=None, query_strategy_kwargs=None):
        """
        Performs a query step, which selects a number of samples from the unlabeled pool.
        A query step must be followed by an update step.

        Parameters
        ----------
        num_samples : int
            Number of samples to query.
        x : list-like
            Alternative representation for the samples in the unlabeled pool.
            This is used by some query strategies.

        Returns
        -------
        queried_indices : numpy.ndarray
            List of queried indices (relative to the current unlabeled pool).

        Raises
        ------
        LearnerNotInitializedException
            Thrown when the active learner was not initialized via `initialize_data(...)`.
        ValueError
            Thrown when args or kwargs are not used and consumed.
        """
        if self._label_to_position is None:
            raise LearnerNotInitializedException()

        size = list_length(self.x_train)
        if x is not None and size != list_length(x):
            raise ValueError('Number of rows of alternative representation x must match the train '
                             'set (dim 0).')

        self.mask = np.ones(size, bool)
        self.mask[self.x_indices_labeled] = False
        indices = np.arange(size)

        x = self.x_train if x is None else x
        query_strategy_kwargs = dict() if query_strategy_kwargs is None else query_strategy_kwargs
        self.queried_indices = self.query_strategy.query(self._clf,
                                                         x,
                                                         indices[self.mask],
                                                         self.x_indices_labeled,
                                                         self.y,
                                                         n=num_samples,
                                                         **query_strategy_kwargs)
        return self.queried_indices

    def update(self, y, x_indices_validation=None):
        """
        Performs an update step, which passes the label for each of the previously queried indices.
        An update step must be preceded by a query step. At the end of the update step the
        current model is retrained using all available labels.

        Parameters
        ----------
        y : list of int or numpy.ndarray
            Labels provided in response to the previous query.
            Each label at index i corresponds to the instance x[i].
        x_indices_validation : numpy.ndarray
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. Otherwise each classifier that uses a validation set will be responsible
            for creating a validation set.
        """
        if len(self.queried_indices) != len(y):
            raise ValueError('Query-update mismatch: indices queried - {} / labels provided - {}'
                             .format(len(self.queried_indices), len(y)
                                     ))

        self.x_indices_labeled = np.concatenate([self.x_indices_labeled, self.queried_indices])
        self._label_to_position = self._build_label_to_position_index()

        if self.x_indices_labeled.shape[0] != np.unique(self.x_indices_labeled).shape[0]:
            raise ValueError('Duplicate indices detected in the labeled pool! '
                             'Please re-examine your query strategy.')

        self.y = np.concatenate([self.y, y])
        self._retrain(x_indices_validation=x_indices_validation)

        self.queried_indices = None
        self.mask = None

    def update_label_at(self, x_index, y, retrain=False, x_indices_validation=None):
        """
        Updates the label for the given x_index (with regard to `self.x_train`).

        Notes
        -----
        After adding labels the current model might not reflect the labeled data anymore.
        You should consider if a retraining is necessary when using this operation.
        Since retraining is often time-consuming, `retrain` is set to `False` by default.

        Parameters
        ----------
        x_index : int
            Label index for the label to be updated.
        y : int
            New label.
        retrain : bool
            Retrains the model after the update if True.
        x_indices_validation : numpy.ndarray
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. Otherwise each classifier that uses a validation set will be responsible
            for creating a validation set.
        """
        position = self._label_to_position[x_index]
        self.y[position] = y

        if retrain:
            self._retrain(x_indices_validation=x_indices_validation)

    def remove_label_at(self, x_index, retrain=False, x_indices_validation=None):
        """
        Removes the labeling for the given x_index (with regard to `self.x_train`).

        Notes
        -----
        After removing labels the current model might not reflect the labeled data anymore.
        You should consider if a retraining is necessary when using this operation.
        Since retraining is often time-consuming, `retrain` is set to `False` by default.

        Parameters
        ----------
        x_index : int
            Label index for the label to be removed.
        retrain : bool
            Retrains the model after the removal if True.
        x_indices_validation : numpy.ndarray
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. Otherwise each classifier that uses a validation set will be responsible
            for creating a validation set.
        """

        position = self._label_to_position[x_index]
        self.y = np.delete(self.y, position)
        self.x_indices_labeled = np.delete(self.x_indices_labeled, position)

        if retrain:
            self._retrain(x_indices_validation=x_indices_validation)

    def save(self, file):
        """
        Serializes the current active learner object into a single file for later re-use.

        Parameters
        ----------
        file : str or path or file
            Serialized output file to be written.
        """
        if isinstance(file, (str, Path)):
           with open(file, 'wb+') as f:
                self._save(f)
        else:
            self._save(file)

    def _save(self, file_handle):
        import dill as pickle
        pickle.dump(version, file_handle)
        pickle.dump(self, file_handle)

    @classmethod
    def load(cls, file):
        """
        Deserializes a serialized active learner.

        Parameters
        ----------
        file : str or path or file
            File to be loaded.
        """
        if isinstance(file, (str, Path)):
            with open(file, 'rb') as f:
                return cls._load(f)
        else:
            return cls._load(file)

    @classmethod
    def _load(self, file_handle):
        import dill as pickle
        _ = pickle.load(file_handle)  # version, will be used in the future
        return pickle.load(file_handle)

    @property
    def classifier(self):
        return self._clf

    @property
    def query_strategy(self):
        return self._query_strategy

    def _retrain(self, x_indices_validation=None):
        if self._clf is None or not self.incremental_training:
            if hasattr(self, '_clf'):
                del self._clf
            self._clf = self._clf_factory.new()

        x = self.x_train[self.x_indices_labeled]

        if x_indices_validation is None:
            self._clf.fit(x)
        else:
            indices = np.arange(self.x_indices_labeled.shape[0])
            mask = np.isin(indices, x_indices_validation)
            if isinstance(self.x_train, list):
                x_train = [x[i] for i in indices[~mask]]
                x_valid = [x[i] for i in indices[mask]]
            else:
                x_train = x[indices[~mask]]
                x_valid = x[indices[mask]]

            self._clf.fit(x_train, validation_set=x_valid)

    def _build_label_to_position_index(self):
        return dict({
            x_index: position for position, x_index in enumerate(self.x_indices_labeled)
        })
