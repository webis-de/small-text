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


def get_splits(train_set, validation_set, weights=None, multi_label=False, validation_set_size=0.1):
    """Helper method to ensure that a validation set is available after calling this method.
    This is only necessary when the previous code did not select a validation set prior to this,
    otherwise the passed `validation_set` variable is not None and no action is necessary here.

    If a split is necessary, stratified sampling is used in the single-label case,
    and random sampling is used in the multi-label case.

    Parameters
    ----------
    train_set : Dataset
        Training set.
    validation_set : Dataset
        Validation set.
    weights : np.ndarray[np.float32] or None, default=None
        Sample weights or None.
    multi_label : bool, default=False
        Indicates if the splits are for a multi-label problem.
    validation_set_size : float, default=0.1
        Specifies the size of the validation set (as a percentage of the training set). Only
        used if a new split is created.

    Returns
    -------
    sub_train : Dataset
        A subset used for training. Defaults to `train_set` if `validation_set` is not `None`.
    sub_valid : Dataset
        A subset used for validation. Defaults to `validation_set` is
    """
    has_validation_set = validation_set is not None
    if has_validation_set:
        indices_train = np.arange(len(train_set))
        result = train_set, validation_set
    else:
        if multi_label:
            # note: this is not an optimal multi-label strategy right now
            indices_train, indices_valid = split_data(train_set,
                                                      y=train_set.y.indices,
                                                      strategy='random',
                                                      validation_set_size=validation_set_size,
                                                      return_indices=True)
        else:
            indices_train, indices_valid = split_data(train_set,
                                                      y=train_set.y,
                                                      strategy='stratified',
                                                      validation_set_size=validation_set_size,
                                                      return_indices=True)
        result = train_set[indices_train], train_set[indices_valid]

    if weights is not None:
        result += (weights,) if not has_validation_set else (weights[indices_train],)

    return result
