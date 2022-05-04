from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import normalize

from small_text.base import check_optional_dependency
from small_text.query_strategies.exceptions import EmptyPoolException, PoolExhaustedException
from small_text.utils.context import build_pbar_context


class QueryStrategy(ABC):
    """Abstract base class for Query Strategies."""

    @abstractmethod
    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        """
        Queries instances from the unlabeled pool.

        A query selects instances from the unlabeled pool.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.
        n : int
            Number of samples to query.

        Returns
        -------
        indices : numpy.ndarray
            Indices relative to `dataset` which were selected.
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

    def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)
        return np.random.choice(indices_unlabeled, size=n, replace=False)

    def __str__(self):
        return 'RandomSampling()'


class ConfidenceBasedQueryStrategy(QueryStrategy):
    """A base class for confidence-based querying.

    To use this class, create a subclass and implement `get_confidence()`.
    """

    def __init__(self, lower_is_better=False):
        self.lower_is_better = lower_is_better
        self.scores_ = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        confidence = self.score(clf, dataset, indices_unlabeled, indices_labeled, y)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_partitioned = np.argpartition(confidence[indices_unlabeled], n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_partitioned])

    def score(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        """Assigns a confidence score to each instance.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.

        Returns
        -------
        confidence : np.ndarray[float]
            Array of confidence scores in the shape (n_samples, n_classes).
            If `self.lower_is_better` the confiden values are flipped to negative so that
            subsequent methods do not need to differentiate maximization/minimization.
        """

        confidence = self.get_confidence(clf, dataset, indices_unlabeled, indices_labeled, y)
        self.scores_ = confidence
        if not self.lower_is_better:
            confidence = -confidence

        return confidence

    @abstractmethod
    def get_confidence(self, clf, dataset, indices_unlabeled, indices_labeled, y):
        """Computes a confidence score for each of the given instances.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : small_text.data.datasets.Dataset
            A text dataset.
        indices_unlabeled : np.ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : np.ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : np.ndarray[int] or csr_matrix
            List of labels where each label maps by index position to `indices_labeled`.
        Returns
        -------
        confidence : ndarray[float]
            Array of confidence scores in the shape (n_samples, n_classes).
        """
        pass

    def __str__(self):
        return 'ConfidenceBasedQueryStrategy()'


class BreakingTies(ConfidenceBasedQueryStrategy):
    """Selects instances which have a small margin between their most likely and second
    most likely predicted class [LUO05]_.
    """

    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        return np.apply_along_axis(lambda x: self._best_versus_second_best(x), 1, proba)

    @staticmethod
    def _best_versus_second_best(proba):
        ind = np.argsort(proba)
        return proba[ind[-1]] - proba[ind[-2]]

    def __str__(self):
        return 'BreakingTies()'


class LeastConfidence(ConfidenceBasedQueryStrategy):
    """Selects instances with the least prediction confidence (regarding the most likely class)
    [LG94]_."""

    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        return np.amax(proba, axis=1)

    def __str__(self):
        return 'LeastConfidence()'


class PredictionEntropy(ConfidenceBasedQueryStrategy):
    """Selects instances with the largest prediction entropy [HOL08]_."""
    def __init__(self):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        return np.apply_along_axis(lambda x: entropy(x), 1, proba)

    def __str__(self):
        return 'PredictionEntropy()'


class SubsamplingQueryStrategy(QueryStrategy):
    """A decorator that first subsamples randomly from the unlabeled pool and then applies
    the `base_query_strategy` on the sampled subset.
    """
    def __init__(self, base_query_strategy, subsample_size=4096):
        """
        Parameters
        ----------
        base_query_strategy : QueryStrategy
            Base query strategy to which the querying is being delegated after subsampling.
        subsample_size : int, default=4096
            Size of the subsampled set.
        """
        self.base_query_strategy = base_query_strategy
        self.subsample_size = subsample_size

        self.subsampled_indices_ = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        subsampled_indices = np.random.choice(indices_unlabeled,
                                              self.subsample_size,
                                              replace=False)
        subset = dataset[np.concatenate([subsampled_indices, indices_labeled])]
        subset_indices_unlabeled = np.arange(self.subsample_size)
        subset_indices_labeled = np.arange(self.subsample_size,
                                           self.subsample_size + indices_labeled.shape[0])

        indices = self.base_query_strategy.query(clf,
                                                 subset,
                                                 subset_indices_unlabeled,
                                                 subset_indices_labeled,
                                                 y,
                                                 n=n)

        self.subsampled_indices_ = indices

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
    """A base class for embedding-based query strategies.

    To use this class, create a subclass and implement `sample()`.
    """
    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar='tqdm',
              embeddings=None, embed_kwargs=dict()):
        self._validate_query_input(indices_unlabeled, n)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_subset_all = np.concatenate([indices_unlabeled, indices_labeled])

        if embeddings is not None:
            proba = None
        else:
            try:
                embeddings, proba = clf.embed(dataset[indices_subset_all],
                                              return_proba=True, pbar=pbar, **embed_kwargs) \
                    if embeddings is None else embeddings

            except TypeError as e:
                if 'got an unexpected keyword argument \'return_proba\'' in e.args[0]:
                    embeddings = clf.embed(dataset[indices_subset_all], pbar=pbar,
                                           **embed_kwargs) if embeddings is None else embeddings
                    proba = None
                else:
                    raise e

        subset = dataset[indices_subset_all]
        subset_indices_unlabeled = np.arange(indices_unlabeled.shape[0])
        subset_indices_labeled = np.arange(indices_unlabeled.shape[0],
                                           indices_unlabeled.shape[0] + indices_labeled.shape[0])

        sampled_indices = self.sample(clf, subset, subset_indices_unlabeled, subset_indices_labeled,
                                      y, n, embeddings, embeddings_proba=proba)

        return np.array([indices_subset_all[i] for i in sampled_indices])

    @abstractmethod
    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        """Samples from the given embeddings.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A text classifier.
        dataset : Dataset
            A text dataset.
        indices_unlabeled : ndarray[int]
            Indices (relative to `dataset`) for the unlabeled data.
        indices_labeled : ndarray[int]
            Indices (relative to `dataset`) for the labeled data.
        y : ndarray[int]
            List of labels where each label maps by index position to `indices_labeled`.
        dataset : ndarray
            Instances for which the score should be computed.
        n : int
            Number of instances to sample.
        embeddings_proba : ndarray, default=None
            Class probabilities for each embedding in embeddings.

        Returns
        -------
        indices : ndarray[int]
            A numpy array of selected indices (relative to `indices_unlabeled`).
        """
        pass

    def __str__(self):
        return 'EmbeddingBasedQueryStrategy()'


class EmbeddingKMeans(EmbeddingBasedQueryStrategy):
    """This is a generalized version of BERT-K-Means [YLB20]_, which is applicable to any kind
    of dense embedding, regardless of the classifier.
    """

    def __init__(self, normalize=True):
        """
        Parameters
        ----------
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        """
        self.normalize = normalize

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        """Samples from the given embeddings.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A classifier.
        dataset : Dataset
            A dataset.
        indices_unlabeled : ndarray
            Indices (relative to `x`) for the unlabeled data.
        indices_labeled : ndarray
            Indices (relative to `x`) for the labeled data.
        y : ndarray or list of int
            List of labels where each label maps by index position to `indices_labeled`.
        dataset : ndarray
            Instances for which the score should be computed.
        embeddings : ndarray
            Embeddings for each sample in x.

        Returns
        -------
        indices : ndarray
            A numpy array of selected indices (relative to `indices_unlabeled`).
        """
        from sklearn.cluster import KMeans

        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        km = KMeans(n_clusters=n)
        km.fit(embeddings[indices_unlabeled])

        indices = self._get_nearest_to_centers(km.cluster_centers_,
                                               embeddings[indices_unlabeled],
                                               normalized=self.normalize)

        # fall back to an iterative version if one or more vectors are most similar
        # to multiple cluster centers
        if np.unique(indices).shape[0] < n:
            indices = self._get_nearest_to_centers_iterative(km.cluster_centers_,
                                                             embeddings[indices_unlabeled],
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


class ContrastiveActiveLearning(EmbeddingBasedQueryStrategy):
    """Contrastive Active Learning [MVB+21]_ selects instances whose k-nearest neighbours
    exhibit the largest mean Kullback-Leibler divergence."""

    def __init__(self, k=10, embed_kwargs=dict(), normalize=True, batch_size=100, pbar='tqdm'):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbours whose KL divergence is considered.
        embed_kwargs : dict
            Embedding keyword args which are passed to `clf.embed()`.
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        batch_size : int, default=100
            Batch size which is used to process the embeddings.
        pbar : 'tqdm' or None, default='tqdm'
            Displays a progress bar if 'tqdm' is passed.
        """
        self.embed_kwargs = embed_kwargs
        self.normalize = normalize
        self.k = k
        self.batch_size = batch_size
        self.pbar = pbar

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar='tqdm',
              embeddings=None, embed_kwargs=dict()):

        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n,
                             embed_kwargs=self.embed_kwargs, pbar=self.pbar)

    def sample(self, _clf, dataset, indices_unlabeled, _indices_labeled, _y, n, embeddings,
               embeddings_proba=None):
        from sklearn.neighbors import NearestNeighbors

        if embeddings_proba is None:
            raise ValueError('Error: embeddings_proba is None. '
                             'This strategy requires a classifier whose embed() method '
                             'supports the return_proba kwarg.')

        if self.normalize:
            embeddings = normalize(embeddings, axis=1)

        nn = NearestNeighbors(n_neighbors=n)
        nn.fit(embeddings)

        return self._contrastive_active_learning(dataset, embeddings, embeddings_proba,
                                                 indices_unlabeled, nn, n)

    def _contrastive_active_learning(self, dataset, embeddings, embeddings_proba,
                                     indices_unlabeled, nn, n):
        from scipy.special import rel_entr

        scores = []

        embeddings_unlabelled_proba = embeddings_proba[indices_unlabeled]
        embeddings_unlabeled = embeddings[indices_unlabeled]

        num_batches = int(np.ceil(len(dataset) / self.batch_size))
        offset = 0
        for batch_idx in np.array_split(np.arange(indices_unlabeled.shape[0]), num_batches,
                                        axis=0):

            nn_indices = nn.kneighbors(embeddings_unlabeled[batch_idx],
                                       n_neighbors=self.k,
                                       return_distance=False)

            kl_divs = np.apply_along_axis(lambda v: np.mean([
                rel_entr(embeddings_proba[i], embeddings_unlabelled_proba[v])
                for i in nn_indices[v - offset]]),
                0,
                batch_idx[None, :])

            scores.extend(kl_divs.tolist())
            offset += batch_idx.shape[0]

        scores = np.array(scores)
        indices = np.argpartition(-scores, n)[:n]

        return indices

    def __str__(self):
        return f'ContrastiveActiveLearning(k={self.k}, ' \
               f'embed_kwargs={str(self.embed_kwargs)}, ' \
               f'normalize={self.normalize})'


class DiscriminativeActiveLearning(QueryStrategy):
    """Discriminative Active Learning [GS19]_ learns to differentiate between the labeled and
    unlabeled pool and selects the instances that are most likely to belong to the unlabeled pool.
    """

    LABEL_LABELED_POOL = 0
    """Label index for the labeled class in the discriminative classification."""

    LABEL_UNLABELED_POOL = 1
    """Label index for the unlabeled class in the discriminative classification."""

    def __init__(self, classifier_factory, num_iterations, unlabeled_factor=10, pbar='tqdm'):
        """
        classifier_factory : small_text.
            Classifier factory which is used for the discriminative classifiers.
        num_iterations : int
            Number of iterations for the discriminiative training.
        unlabeled_factor : int, default=10
            The ratio of "unlabeled pool" instances to "labeled pool" instances in the
            discriminative training.
        pbar : 'tqdm' or None, default='tqdm'
            Displays a progress bar if 'tqdm' is passed.
        """
        self.classifier_factory = classifier_factory
        self.num_iterations = num_iterations

        self.unlabeled_factor = unlabeled_factor
        self.pbar = pbar

        self.clf_ = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        query_sizes = self._get_query_sizes(self.num_iterations, n)
        indices = self.discriminative_active_learning(dataset, indices_unlabeled, indices_labeled,
                                                      query_sizes)

        return indices

    def discriminative_active_learning(self, dataset, indices_unlabeled, indices_labeled,
                                       query_sizes):

        indices = np.array([], dtype=indices_labeled.dtype)

        indices_unlabeled_copy = np.copy(indices_unlabeled)
        indices_labeled_copy = np.copy(indices_labeled)

        with build_pbar_context(len(query_sizes)) as pbar:
            for q in query_sizes:
                indices_most_confident = self._train_and_get_most_confident(dataset,
                                                                            indices_unlabeled_copy,
                                                                            indices_labeled_copy,
                                                                            q)

                indices = np.append(indices, indices_unlabeled_copy[indices_most_confident])
                indices_labeled_copy = np.append(indices_labeled_copy,
                                                 indices_unlabeled_copy[indices_most_confident])
                indices_unlabeled_copy = np.delete(indices_unlabeled_copy, indices_most_confident)
                pbar.update(1)

        return indices

    @staticmethod
    def _get_query_sizes(num_iterations, n):

        if num_iterations > n:
            raise ValueError('num_iterations cannot be greater than the query_size n')

        query_size = int(n / num_iterations)
        query_sizes = [query_size if i < num_iterations - 1
                       else n - (num_iterations - 1) * query_size
                       for i, _ in enumerate(range(num_iterations))]

        return query_sizes

    def _train_and_get_most_confident(self, ds, indices_unlabeled, indices_labeled, q):

        if self.clf_ is not None:
            del self.clf_

        clf = self.classifier_factory.new()

        num_unlabeled = min(indices_labeled.shape[0] * self.unlabeled_factor,
                            indices_unlabeled.shape[0])

        indices_unlabeled_sub = np.random.choice(indices_unlabeled,
                                                 num_unlabeled,
                                                 replace=False)

        ds_discr = DiscriminativeActiveLearning.get_relabeled_copy(ds,
                                                                   indices_unlabeled_sub,
                                                                   indices_labeled)

        self.clf_ = clf.fit(ds_discr)

        proba = clf.predict_proba(ds[indices_unlabeled])
        proba = proba[:, self.LABEL_UNLABELED_POOL]

        # return instances which most likely belong to the "unlabeled" class (higher is better)
        return np.argpartition(-proba, q)[:q]

    @staticmethod
    def get_relabeled_copy(dataset, indices_unlabeled_sub, indices_labeled):

        if dataset.is_multi_label:
            raise NotImplementedError('Only single-label datasets are supported')

        indices_train = np.append(indices_unlabeled_sub, indices_labeled)
        ds_sub = dataset[indices_train].clone()

        # relabel dataset as "unlabeled" (pool) and "labeled" (pool)
        ds_sub.y = np.array(
            [DiscriminativeActiveLearning.LABEL_UNLABELED_POOL] * indices_unlabeled_sub.shape[0] +
            [DiscriminativeActiveLearning.LABEL_LABELED_POOL] * indices_labeled.shape[0]
        )

        return ds_sub

    def __str__(self):
        return f'DiscriminativeActiveLearning(classifier_factory={str(self.classifier_factory)}, ' \
               f'num_iterations={self.num_iterations}, unlabeled_factor={self.unlabeled_factor})'


class SEALS(QueryStrategy):
    """Similarity Search for Efficient Active Learning and Search of Rare Concepts (SEALS)
    improves the computational efficiency of active learning by presenting a reduced subset
    of the unlabeled pool to a base strategy [CCK+21]_.

    This method is to be applied in conjunction with a base query strategy. SEALS selects a
    subset of the unlabeled pool by selecting the `k` nearest neighbours of the current labeled
    pool.

    If the size of the unlabeled pool falls below the given `k`, this implementation will
    not select a subset anymore and will just delegate to the base strategy instead.

    .. note ::
       This strategy requires the optional dependency `hnswlib`.
    """
    def __init__(self, base_query_strategy, k=100, hnsw_kwargs=dict(), embed_kwargs=dict(),
                 normalize=True):
        """
        base_query_strategy : small_text.query_strategy.QueryStrategy
            A base query strategy which operates on the subset that is selected by SEALS.
        k : int, default=100
            Number of nearest neighbors that will be selected.
        hnsw_kwargs : dict(), default=dict()
            Kwargs which will be passed to the underlying hnsw index.
            Check the `hnswlib github repository <https://github.com/nmslib/hnswlib>`_ on details
            for the parameters `space`, `ef_construction`, `ef`, and `M`.
        embed_kwargs : dict, default=dict()
            Kwargs that will be passed to the embed() method.
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        """
        check_optional_dependency('hnswlib')

        self.base_query_strategy = base_query_strategy
        self.k = k
        self.hnsw_kwargs = hnsw_kwargs
        self.embed_kwargs = embed_kwargs
        self.normalize = normalize

        self.nn = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar='tqdm'):

        if self.k > indices_unlabeled.shape[0]:
            return self.base_query_strategy.query(clf, dataset, indices_unlabeled, indices_labeled,
                                                  y, n=n)

        indices_subset = self.get_subset_indices(clf,
                                                 dataset,
                                                 indices_unlabeled,
                                                 indices_labeled,
                                                 pbar=pbar)
        return self.base_query_strategy.query(clf, dataset, indices_subset, indices_labeled, y, n=n)

    def get_subset_indices(self, clf, dataset, indices_unlabeled, indices_labeled, pbar='tqdm'):
        if self.nn is None:
            self.embeddings = clf.embed(dataset, pbar=pbar)
            if self.normalize:
                self.embeddings = normalize(self.embeddings, axis=1)

            self.nn = self.initialize_index(self.embeddings, indices_unlabeled, self.hnsw_kwargs)
            self.indices_unlabeled = set(indices_unlabeled)
        else:
            recently_removed_elements = self.indices_unlabeled - set(indices_unlabeled)
            for el in recently_removed_elements:
                self.nn.mark_deleted(el)
            self.indices_unlabeled = set(indices_unlabeled)

        indices_nn, _ = self.nn.knn_query(self.embeddings[indices_labeled], k=self.k)
        indices_nn = np.unique(indices_nn.astype(int).flatten())

        return indices_nn

    @staticmethod
    def initialize_index(embeddings, indices_unlabeled, hnsw_kwargs):
        import hnswlib

        space = hnsw_kwargs.get('space', 'l2')
        ef_construction = hnsw_kwargs.get('ef_construction', 200)
        m = hnsw_kwargs.get('M', 64)
        ef = hnsw_kwargs.get('ef', 200)

        index = hnswlib.Index(space=space, dim=embeddings.shape[1])
        index.init_index(max_elements=embeddings.shape[0],
                         ef_construction=ef_construction,
                         M=m)
        index.add_items(embeddings[indices_unlabeled], indices_unlabeled)
        index.set_ef(ef)

        return index

    def __str__(self):
        return f'SEALS(base_query_strategy={str(self.base_query_strategy)}, ' \
               f'k={self.k}, embed_kwargs={self.embed_kwargs}, normalize={self.normalize})'
