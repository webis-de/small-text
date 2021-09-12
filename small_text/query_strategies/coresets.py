import numpy as np

from small_text.query_strategies.strategies import QueryStrategy


def lightweight_coreset(n, x, x_mean, proba=None):
    """

    Parameters
    ----------
    n : int
        Coreset size.
    x : np.ndarray
        2D array in which each row represents a sample.
    x_mean : np.ndarray
        Elementwise mean over the columns of `x`.
    proba : np.ndarray

    Returns
    -------
    indices : numpy.ndarray
        Indices relative to `x`.

    References
    ----------
    .. [BAC18] Olivier Bachem, Mario Lucic, and Andreas Krause. 2018.
               Scalable k -Means Clustering via Lightweight Coresets.
               In Proceedings of the 24th ACM SIGKDD International Conference on
                 Knowledge Discovery & Data Mining (KDD '18).
               Association for Computing Machinery, New York, NY, USA, 1119–1127.
    """
    if n > x.shape[0]:
        raise ValueError(f'n (n={n}) is greater the number of available samples (num_samples={x.shape[0]})')

    dists = 1 - x.dot(x_mean) / (np.linalg.norm(x, axis=1) * np.linalg.norm(x_mean))
    dists = np.square(dists)

    sum_dists = dists.sum()

    if proba is None:
        uniform = 0.5 * 1 / x.shape[0]
        proba = uniform + 0.5 * dists / sum_dists
    else:
        proba = 0.5 * proba / proba.sum() + 0.5 * dists / sum_dists

    proba = proba / np.linalg.norm(proba, ord=1)

    return np.random.choice(np.arange(x.shape[0]), n, replace=False, p=proba)


class LightweightCoreset(QueryStrategy):
    """Selects instances using the least coreset method _[BAC18].

    Parameters
    ----------
    """
    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10, pbar=None, embed_kwargs=None,
              embeddings=None, normalize=False):
        self._validate_query_input(x_indices_unlabeled, n)

        embed_kwargs = dict() if embed_kwargs is None else embed_kwargs
        embeddings = embeddings[x_indices_unlabeled] if embeddings is not None \
            else clf.embed(x[x_indices_unlabeled], pbar=pbar, **embed_kwargs)

        embeddings_mean = np.mean(embeddings, axis=0)
        if normalize:
            embeddings = normalize(embeddings)
            embeddings_mean = normalize(embeddings_mean)

        embeddings_mean = embeddings_mean.ravel()

        indices = lightweight_coreset(n, embeddings, embeddings_mean)

        return np.array([x_indices_unlabeled[i] for i in indices])

    def __str__(self):
        return 'LightweightCoreset()'
