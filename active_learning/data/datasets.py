import numpy as np

from abc import ABC

from active_learning.data.sampling import stratified_sampling, balanced_sampling


class Dataset(ABC):

    @property
    def x(self):
        pass

    @x.setter
    def x(self, x_new):
        pass

    @property
    def y(self):
        pass

    @y.setter
    def y(self, y_new):
        pass

    @property
    def target_labels(self):
        pass

    @target_labels.setter
    def target_labels(self, target_labels):
        pass


class SklearnDataSet(Dataset):

    def __init__(self, x, y, target_labels=None):
        """
        Parameters
        ----------
        x : numpy.ndarray or scipy.sparse.csr_matrix
            Dense or sparse feature matrix.
        y : list of int
            List of labels where each label belongs to the features of the respective row.
        target_labels : list of int or None
            List of target_labels. Will be inferred from `y` if None is passed.
        """
        self._x = x
        self._y = np.array(y)

        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = np.array(target_labels)
        else:
            self.track_target_labels = True
            self._infer_target_labels(self._y)

    def _infer_target_labels(self, y):
        self.target_labels = np.unique(y)

    @property
    def x(self):
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
    def target_labels(self):
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        # TODO: how to handle existing labels that are outside this set
        self._target_labels = target_labels

    def __getitem__(self, item):
        if isinstance(item, list) or isinstance(item, np.ndarray) or isinstance(item, slice):
            return SklearnDataSet(self._x[item], np.array(self._y)[item])

        ds = SklearnDataSet(self._x[item], self._y[item])
        if len(ds._x.shape) == 1:
            ds._x = ds._x[np.newaxis, :]

        return ds

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
