import numpy as np

from active_learning.utils.data import list_length
from active_learning.data.sampling import stratified_sampling, balanced_sampling


def random_initialization(x, n_samples=10):
    """Randomly draws from the given dataset x.

    Parameters
    ----------
    x :
        A supported dataset.
    n_samples :  int
        Number of samples to draw.

    Returns
    -------
    indices : np.array[int]
        Numpy array containing indices relative to x.
    """
    return np.random.choice(list_length(x), size=n_samples, replace=False)


def random_initialization_stratified(y, n_samples=10):
    return stratified_sampling(y, n_samples=n_samples)


def random_initialization_balanced(y, n_samples=10):
    return balanced_sampling(y, n_samples=n_samples)
