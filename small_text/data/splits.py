import numpy as np
from small_text.data.sampling import stratified_sampling, balanced_sampling


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
