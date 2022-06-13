import numpy as np

from scipy.sparse import csr_matrix

from small_text.utils.data import list_length
from small_text.data.sampling import balanced_sampling
from small_text.data.sampling import multilabel_stratified_subsets_sampling
from small_text.data.sampling import stratified_sampling


def random_initialization(x, n_samples=10):
    """Randomly draws a subset from the given dataset.

    Parameters
    ----------
    x : Dataset
        A dataset.
    n_samples : int, default=10
        Number of samples to draw.

    Returns
    -------
    indices : np.ndarray[int]
        Indices relative to x.
    """
    return np.random.choice(list_length(x), size=n_samples, replace=False)


def random_initialization_stratified(y, n_samples=10, multilabel_strategy='labelsets'):
    """Randomly draws a subset stratified by class labels.

    Parameters
    ----------
    y : np.ndarray[int] or csr_matrix
        Labels to be used for stratification.
    n_samples :  int
        Number of samples to draw.
    multilabel_strategy : {'labelsets'}, default='labelsets'
        The multi-label strategy to be used in case of a multi-label labeling.
        This is only used if `y` is of type csr_matrix.

    Returns
    -------
    indices : np.ndarray[int]
        Indices relative to y.

    See Also
    --------
    small_text.data.sampling.multilabel_stratified_subsets_sampling : Details on the `labelsets`
        multi-label strategy.
    """
    if isinstance(y, csr_matrix):
        if multilabel_strategy == 'labelsets':
            return multilabel_stratified_subsets_sampling(y, n_samples=n_samples)
        else:
            raise ValueError(f'Invalid multilabel_strategy: {multilabel_strategy}')
    else:
        return stratified_sampling(y, n_samples=n_samples)


def random_initialization_balanced(y, n_samples=10):
    """Randomly draws a subset which is (approximately) balanced in the distribution
    of its class labels.

    Parameters
    ----------
    y : np.ndarray[int] or csr_matrix
        Labels to be used for balanced sampling.
    n_samples : int, default=10
        Number of samples to draw.

    Returns
    -------
    indices : np.ndarray[int]
        Indices relative to y.

    Notes
    -----
    This is only applicable to single-label classification.
    """
    if isinstance(y, csr_matrix):
        raise NotImplementedError()
    else:
        return balanced_sampling(y, n_samples=n_samples)
