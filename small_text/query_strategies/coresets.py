import warnings

import numpy as np

from sklearn.metrics import pairwise_distances
from small_text.query_strategies.strategies import EmbeddingBasedQueryStrategy


_DISTANCE_METRICS = ['cosine', 'euclidean']


def _check_coreset_size(x, n):
    if n > x.shape[0]:
        raise ValueError(f'n (n={n}) is greater the number of available samples (num_samples={x.shape[0]})')


def _cosine_distance(a, b, normalized=False):
    sim = np.matmul(a, b.T)
    if not normalized:
        sim = sim / np.dot(np.linalg.norm(a, axis=1)[:, np.newaxis],
                           np.linalg.norm(b, axis=1)[np.newaxis, :])
    return np.arccos(sim) / np.pi


def _euclidean_distance(a, b, normalized=False):
    _ = normalized
    return pairwise_distances(a, b, metric='euclidean')


def greedy_coreset(x, indices_unlabeled, indices_labeled, n, distance_metric='cosine',
                   batch_size=100, normalized=False):
    """Computes a greedy coreset [SS17]_ over `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        A matrix of row-wise vector representations.
    indices_unlabeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    indices_labeled : np.ndarray
        Indices (relative to `dataset`) for the unlabeled data.
    n : int
        Size of the coreset (in number of instances).
    distance_metric : {'cosine', 'euclidean'}
        Distance metric to be used.
    batch_size : int
        Batch size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [SS17] Ozan Sener and Silvio Savarese. 2017.
       Active Learning for Convolutional Neural Networks: A Core-Set Approach.
       In International Conference on Learning Representations 2018 (ICLR 2018).
    """
    _check_coreset_size(x, n)

    num_batches = int(np.ceil(x.shape[0] / batch_size))
    ind_new = []

    if distance_metric == 'cosine':
        dist_func = _cosine_distance
    elif distance_metric == 'euclidean':
        dist_func = _euclidean_distance
    else:
        raise ValueError(f'Invalid distance metric: {distance_metric}. '
                         f'Possible values: {_DISTANCE_METRICS}')

    for _ in range(n):
        indices_s = np.concatenate([indices_labeled, ind_new]).astype(np.int64)
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x[indices_unlabeled], num_batches, axis=0):

            dist = dist_func(batch, x[indices_s], normalized=normalized)

            sims_batch = np.amin(dist, axis=1)
            dists = np.append(dists, sims_batch)

        dists[ind_new] = -np.inf
        index_new = np.argmax(dists)

        ind_new.append(index_new)

    return np.array(ind_new)


class GreedyCoreset(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a greedy coreset [SS17]_ over document embeddings.
    """
    def __init__(self, distance_metric='euclidean', normalize=True, batch_size=100):
        """
        Parameters
        ----------
        distance_metric : {'cosine', 'euclidean'}
             Distance metric to be used.

             .. versionadded:: 1.2.0
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        batch_size : int
            Batch size used for computing document distances.


        .. note::

           The default distance metric before v1.2.0 used to be cosine distance.

        .. seealso::

           Function :py:func:`.greedy_coreset`
              Docstrings of the underlying :py:func:`greedy_coreset` method.
        """
        if distance_metric not in set(_DISTANCE_METRICS):
            raise ValueError(f'Invalid distance metric: {distance_metric}. '
                             f'Possible values: {_DISTANCE_METRICS}')

        if distance_metric != 'cosine':
            warnings.warn('Default distance metric has changed from "cosine" '
                          'to "euclidean" in v1.2.0. This warning will disappear in '
                          'v2.0.0.')

        self.distance_metric = distance_metric
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)
        return greedy_coreset(embeddings, indices_unlabeled, indices_labeled, n,
                              distance_metric=self.distance_metric, normalized=self.normalize)

    def __str__(self):
        return f'GreedyCoreset(distance_metric={self.distance_metric}, ' \
            f'normalize={self.normalize}, batch_size={self.batch_size})'


def lightweight_coreset(x, x_mean, n, normalized=False, proba=None):
    """Computes a lightweight coreset [BLK18]_ of `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray
        2D array in which each row represents a sample.
    x_mean : np.ndarray
        Elementwise mean over the columns of `x`.
    n : int
        Coreset size.
    normalized : bool
        If `True` the data `x` is assumed to be normalized,
        otherwise it will be normalized where necessary.
    proba : np.ndarray or None
        A probability distribution over `x`, which makes up half of the probability mass
        of the sampling distribution. If `proba` is not `None` a uniform distribution is used.

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.
    """
    _check_coreset_size(x, n)

    sim = x.dot(x_mean)
    if not normalized:
        sim = sim / (np.linalg.norm(x, axis=1) * np.linalg.norm(x_mean))

    dists = np.arccos(sim) / np.pi
    dists = np.square(dists)

    sum_dists = dists.sum()

    if proba is None:
        uniform = 0.5 * 1 / x.shape[0]
        proba = uniform + 0.5 * dists / sum_dists
    else:
        proba = 0.5 * proba / proba.sum() + 0.5 * dists / sum_dists

    proba = proba / np.linalg.norm(proba, ord=1)

    return np.random.choice(np.arange(x.shape[0]), n, replace=False, p=proba)


class LightweightCoreset(EmbeddingBasedQueryStrategy):
    """Selects instances by constructing a lightweight coreset [BLK18]_ over document embeddings.
    """
    def __init__(self, normalize=True):
        """
        Parameters
        ----------
        normalize : bool
            Embeddings will be normalized before the coreset construction if True.
        """
        self.normalize = normalize

    def sample(self, clf, dataset, indices_unlabeled, _indices_labeled, _y, n, embeddings,
               embeddings_proba=None):

        embeddings = embeddings[indices_unlabeled]

        embeddings_mean = np.mean(embeddings, axis=0)
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings)
            embeddings_mean = normalize(embeddings_mean[np.newaxis, :])

        embeddings_mean = embeddings_mean.ravel()

        return lightweight_coreset(embeddings, embeddings_mean, n, normalized=self.normalize)

    def __str__(self):
        return f'LightweightCoreset(normalize={self.normalize})'
