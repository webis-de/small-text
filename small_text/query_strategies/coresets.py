import numpy as np

from small_text.query_strategies.strategies import EmbeddingBasedQueryStrategy


def _check_coreset_size(x, n):
    if n > x.shape[0]:
        raise ValueError(f'n (n={n}) is greater the number of available samples (num_samples={x.shape[0]})')


def greedy_coreset(x, x_indices_unlabeled, x_indices_labeled, n, batch_size=100, normalized=False):
    """Computes a greedy coreset _[SS17] of `x` with size `n`.

    Parameters
    ----------
    x : np.ndarray

    x_indices_unlabeled : np.ndarray

    x_indices_labeled : np.ndarray

    n : int

    batch_size : int

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

    for _ in range(n):
        indices_s = np.concatenate([x_indices_labeled, ind_new]).astype(np.int64)
        sims = np.array([], dtype=np.float32)
        for batch in np.array_split(x[x_indices_unlabeled], num_batches, axis=0):

            sim = np.matmul(batch, x[indices_s].T)
            if not normalized:
                sim = sim / np.dot(np.linalg.norm(batch, axis=1)[:, np.newaxis],
                                   np.linalg.norm(x[indices_s], axis=1)[np.newaxis, :])

            sims_batch = np.amax(sim, axis=1)
            sims = np.append(sims, sims_batch)

        sims[ind_new] = np.inf
        index_new = np.argmin(sims)

        ind_new.append(index_new)

    return np.array(ind_new)


class GreedyCoreset(EmbeddingBasedQueryStrategy):

    def __init__(self, normalize=True, batch_size=100):
        self.normalize = normalize
        self.batch_size = batch_size

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)
        return greedy_coreset(embeddings, indices_unlabeled, indices_labeled, n,
                              normalized=self.normalize)

    def __str__(self):
        return f'GreedyCoreset(normalize={self.normalize}, batch_size={self.batch_size})'


def lightweight_coreset(x, x_mean, n, normalized=False, proba=None):
    """Computes a lightweight coreset _[BAC18] of `x` with size `n`.

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

    References
    ----------
    .. [BAC18] Olivier Bachem, Mario Lucic, and Andreas Krause. 2018.
               Scalable k-Means Clustering via Lightweight Coresets.
               In Proceedings of the 24th ACM SIGKDD International Conference on
                 Knowledge Discovery & Data Mining (KDD '18).
               Association for Computing Machinery, New York, NY, USA, 1119–1127.
    """
    _check_coreset_size(x, n)

    sim = x.dot(x_mean)
    if not normalized:
        sim = sim / (np.linalg.norm(x, axis=1) * np.linalg.norm(x_mean))
    dists = 1 - sim
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
    """Selects instances using the lightweight coreset method _[BAC18].

    Parameters
    ----------
    normalize : bool
        Embeddings are normalized if `True`, otherwise they are left unchanged.
    """
    def __init__(self, normalize=True):
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
