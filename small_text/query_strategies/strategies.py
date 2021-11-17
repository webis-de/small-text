from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import entropy

from small_text.query_strategies.exceptions import EmptyPoolException, PoolExhaustedException


class QueryStrategy(ABC):
    """Abstract base class for Query Strategies."""

    @abstractmethod
    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10):
        """
        A query selects instances from the unlabeled pool.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A classifier.
        x : small_text.data.datasets.Dataset
            A dataset.
        x_indices_unlabeled : list of int
            Indices (relative to `x`) for the unlabeled data.
        x_indices_labeled : list of int
            Indices (relative to `x`) for the labeled data.
        y : list of int
            List of labels where each label maps by index position to `indices_labeled`.
        n : int
            Number of samples to query.

        Returns
        -------
        indices : numpy.ndarray
            Indices relative to `dataset`.
        """
        pass

    @staticmethod
    def _validate_query_input(indices_unlabeled, n):
        if len(indices_unlabeled) == 0:
            raise EmptyPoolException('No unlabeled indices available. Cannot query an empty pool.')

        if n > len(indices_unlabeled):
            raise PoolExhaustedException('Pool exhausted: {} available / {} requested'
                                         .format(len(indices_unlabeled), n))


class RandomSampling(QueryStrategy):
    """Randomly selects instances."""

    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10):
        self._validate_query_input(x_indices_unlabeled, n)
        return np.random.choice(x_indices_unlabeled, size=n, replace=False)

    def __str__(self):
        return 'RandomSampling()'


class ConfidenceBasedQueryStrategy(QueryStrategy):
    """A base class for confidence-based querying. To use this class, create a subclass and
     implement `get_confidence()`.
    """

    def __init__(self, lower_is_better=False):
        self.lower_is_better = lower_is_better
        self.scores_ = None

    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10):
        self._validate_query_input(x_indices_unlabeled, n)

        confidence = self.score(clf, x, x_indices_unlabeled, x_indices_labeled, y)

        if len(x_indices_unlabeled) == n:
            return np.array(x_indices_unlabeled)

        indices_partitioned = np.argpartition(confidence[x_indices_unlabeled], n)[:n]
        return np.array([x_indices_unlabeled[i] for i in indices_partitioned])

    def score(self, clf, dataset, indices_unlabeled, indices_labeled, y):

        confidence = self.get_confidence(clf, dataset, indices_unlabeled, indices_labeled, y)
        self.scores_ = confidence
        if not self.lower_is_better:
            confidence = -confidence

        return confidence

    @abstractmethod
    def get_confidence(self, clf, x, x_indices_unlabeled, x_indices_labeled, y):
        """
        Computes a confidence score for each given instance.

        Parameters
        ----------
        x : ndarray
            Instances for which the confidence should be computed.

        Returns
        -------
        confidence : ndarray
            A 2D numpy array (of type float) in the shape (n_samples, n_classes).
        """
        pass

    def __str__(self):
        return 'ConfidenceBasedQueryStrategy()'


class BreakingTies(ConfidenceBasedQueryStrategy):
    """Selects instances which have a small margin between their most likely and second
    most likely prediction.

    References
    ----------
    .. [LUO05] Tong Luo, Kurt Kramer, Dmitry B. Goldgof, Lawrence O. Hall, Scott Samson,
       Andrew Remsen, and Thomas Hopkins. 2005.
       Active Learning to Recognize Multiple Types of Plankton.
       J. Mach. Learn. Res. 6 (12/1/2005), 589–613.
    """

    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, x, _x_indices_unlabeled, _x_indices_labeled, _y):
        proba = clf.predict_proba(x)
        return np.apply_along_axis(lambda x: self._best_versus_second_best(x), 1, proba)

    @staticmethod
    def _best_versus_second_best(proba):
        ind = np.argsort(proba)
        return proba[ind[-1]] - proba[ind[-2]]

    def __str__(self):
        return 'BreakingTies()'


class LeastConfidence(ConfidenceBasedQueryStrategy):
    """Selects instances with the least prediction confidence (regarding the most likely class) [LG94]_.

    References
    ----------
    .. [LG94] David D. Lewis and William A. Gale.
       A sequential algorithm for training text classifiers.
       In SIGIR’94, 1994, 3-12.
    """

    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, x, _x_indices_unlabeled, _x_indices_labeled, _y):
        proba = clf.predict_proba(x)
        return np.amax(proba, axis=1)

    def __str__(self):
        return 'LeastConfidence()'


class PredictionEntropy(ConfidenceBasedQueryStrategy):
    """Selects instances with the largest prediction entropy [HOL08]_.

    References
    ----------
    .. [HOL08] Alex Holub, Pietro Perona, and Michael C. Burl. 2008.
       Entropy-based active learning for object recognition.
       In 2008 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops,
       IEEE, 1–8.
    """
    def __init__(self):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, x, _x_indices_unlabeled, _x_indices_labeled, _y):
        proba = clf.predict_proba(x)
        return np.apply_along_axis(lambda x: entropy(x), 1, proba)

    def __str__(self):
        return 'PredictionEntropy()'


class SubsamplingQueryStrategy(QueryStrategy):
    """A decorator that first subsamples randomly from the unlabeled pool and then applies
    the `base_query_strategy` on the sampled subset.

    Parameters
    ----------
    base_query_strategy : QueryStrategy
        Base query strategy.
    subsample_size : int
        Size of the subsampled set.
    """
    def __init__(self, base_query_strategy, subsample_size=4096):
        self.base_query_strategy = base_query_strategy
        self.subsample_size = subsample_size

        self.subsampled_indices_ = None

    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10):
        self._validate_query_input(x_indices_unlabeled, n)

        subsampled_indices = np.random.choice(x_indices_unlabeled, self.subsample_size,
                                              replace=False)
        x_sub = x[np.concatenate([subsampled_indices, x_indices_labeled])]
        x_sub_unlabeled = np.arange(self.subsample_size)
        x_sub_labeled = np.arange(self.subsample_size,
                                  self.subsample_size + x_indices_labeled.shape[0])

        indices = self.base_query_strategy.query(clf, x_sub, x_sub_unlabeled, x_sub_labeled, y, n=n)

        self.subsampled_indices_ = x_sub_unlabeled

        return np.array([subsampled_indices[i] for i in indices])

    @property
    def scores_(self):
        if hasattr(self.base_query_strategy, 'scores_'):
            return self.base_query_strategy.scores_[:self.subsample_size]
        return None

    def __str__(self):
        return f'SubsamplingQueryStrategy(base_query_strategy={self.base_query_strategy}, ' \
               f'subsample_size={self.subsample_size})'


class EmbeddingBasedQueryStrategy(QueryStrategy):
    """A base class for embedding-based querying. To use this class, create a subclass and
     implement `sample()`.
    """
    def query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10, pbar='tqdm',
              embeddings=None, embed_kwargs=dict()):
        self._validate_query_input(x_indices_unlabeled, n)

        if len(x_indices_unlabeled) == n:
            return np.array(x_indices_unlabeled)

        if embeddings is not None:
            sampled_indices = self.sample(clf, x, x_indices_unlabeled, x_indices_labeled, y, n,
                                          embeddings)
        else:
            try:
                embeddings, proba = clf.embed(x, return_proba=True, pbar=pbar, **embed_kwargs) \
                    if embeddings is None else embeddings
                sampled_indices = self.sample(clf, x, x_indices_unlabeled, x_indices_labeled,
                                              y, n, embeddings, embeddings_proba=proba)
            except TypeError as e:
                if 'got an unexpected keyword argument \'return_proba\'' in e.args[0]:
                    embeddings = clf.embed(x, pbar=pbar,
                                           **embed_kwargs) if embeddings is None else embeddings
                    sampled_indices = self.sample(clf, x, x_indices_unlabeled, x_indices_labeled, y,
                                                  n, embeddings)
                else:
                    raise e

        return x_indices_unlabeled[sampled_indices]

    @abstractmethod
    def sample(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        """Samples from the given embeddings.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A classifier.
        x : Dataset
            A dataset.
        x_indices_unlabeled : ndarray
            Indices (relative to `x`) for the unlabeled data.
        x_indices_labeled : ndarray
            Indices (relative to `x`) for the labeled data.
        y : ndarray or list of int
            List of labels where each label maps by index position to `indices_labeled`.
        x : ndarray
            Instances for which the score should be computed.
        embeddings : ndarray
            Embeddings for each sample in x.
        embeddings_proba : ndarray or None
            Class probabilities for each embedding in embeddings.

        Returns
        -------
        indices : ndarray
            A numpy array of selected indices (relative to `x_indices_unlabeled`).
        """
        pass

    def __str__(self):
        return 'EmbeddingBasedQueryStrategy()'


class EmbeddingKMeans(EmbeddingBasedQueryStrategy):
    """This is a generalized version of BERT-K-Means  [YLB20]_, which is applicable to any kind
    of dense embedding, regardless of the classifier.

    References
    ----------
    .. [YLB20] Michelle Yuan, Hsuan-Tien Lin, and Jordan Boyd-Graber. 2020.
       Cold-start Active Learning through Self-supervised Language Modeling
       In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)
       Association for Computational Linguistics, 7935–-7948.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def sample(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        """Samples from the given embeddings.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A classifier.
        x : Dataset
            A dataset.
        x_indices_unlabeled : ndarray
            Indices (relative to `x`) for the unlabeled data.
        x_indices_labeled : ndarray
            Indices (relative to `x`) for the labeled data.
        y : ndarray or list of int
            List of labels where each label maps by index position to `indices_labeled`.
        x : ndarray
            Instances for which the score should be computed.
        embeddings : ndarray
            Embeddings for each sample in x.

        Returns
        -------
        indices : ndarray
            A numpy array of selected indices (relative to `x_indices_unlabeled`).
        """
        from sklearn.cluster import KMeans

        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        km = KMeans(n_clusters=n)
        km.fit(embeddings[x_indices_unlabeled])

        indices = self._get_nearest_to_centers(km.cluster_centers_,
                                               embeddings[x_indices_unlabeled],
                                               normalized=self.normalize)

        # fall back to an iterative version if one or more vectors are most similar
        # to multiple cluster centers
        if np.unique(indices).shape[0] < n:
            indices = self._get_nearest_to_centers_iterative(km.cluster_centers_,
                                                             embeddings[x_indices_unlabeled],
                                                             normalized=self.normalize)

        return indices

    @staticmethod
    def _get_nearest_to_centers(centers, vectors, normalized=True):
        sim = EmbeddingKMeans._similarity(centers, vectors, normalized)
        return sim.argmax(axis=1)

    @staticmethod
    def _similarity(centers, vectors, normalized):
        sim = np.matmul(centers, vectors.T)

        if not normalized:
            sim = sim / np.dot(np.linalg.norm(centers, axis=1)[:, np.newaxis],
                               np.linalg.norm(vectors, axis=1)[np.newaxis, :])
        return sim

    @staticmethod
    def _get_nearest_to_centers_iterative(cluster_centers, vectors, normalized=True):

        indices = np.empty(cluster_centers.shape[0], dtype=int)

        for i in range(cluster_centers.shape[0]):
            sim = EmbeddingKMeans._similarity(cluster_centers[None, i], vectors, normalized)
            sim[0, indices[0:i]] = -np.inf
            indices[i] = sim.argmax()

        return indices

    def __str__(self):
        return f'EmbeddingKMeans(normalize={self.normalize})'
