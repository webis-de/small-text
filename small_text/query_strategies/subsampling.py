import numpy as np

from sklearn.preprocessing import normalize

from small_text.base import check_optional_dependency
from small_text.data.sampling import _get_class_histogram
from small_text.query_strategies.strategies import QueryStrategy
from small_text.utils.clustering import init_kmeans_plusplus_safe


class AnchorSubsampling(QueryStrategy):
    """This subsampling strategy is an implementation of AnchorAL [LV24]_.

    AnchorAL performs subsampling with class-specific anchors, which aims to draw class-balanced subset and
    to prevent overfitting on the current decision boundary [LV24]_.

    This method is very extensible regarding the choices of base query strategy and anchor selection,
    but for now the implementation covers the choices described in the original paper.

    .. note ::
       This strategy requires the optional dependency `hnswlib`.

    .. versionadded:: 1.4.0

    """
    def __init__(self, base_query_strategy, subsample_size=500, num_anchors=10, k=50, hnsw_kwargs={}, embed_kwargs={},
                 normalize=True, batch_size=32):
        """
        base_query_strategy : small_text.query_strategy.QueryStrategy
            A base query strategy which operates on the subset that is selected by SEALS.
        subsample_size : int, default=500
            The number of subsamples to be drawn.
        k : int, default=50
            Number of nearest neighbors that will be selected.
        hnsw_kwargs : dict, default=dict{}
            Keyword arguments that will be passed to the underlying hnsw index.
            Check the `hnswlib github repository <https://github.com/nmslib/hnswlib>`_ on details
            for the parameters `space`, `ef_construction`, `ef`, and `M`.
        embed_kwargs : dict, default=dict{}
            Keyword arguments that will be passed to the embed() method.
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        """
        check_optional_dependency('hnswlib')

        self.base_query_strategy = base_query_strategy
        self.subsample_size = subsample_size
        self.num_anchors = num_anchors
        self.k = k
        self.hnsw_kwargs = hnsw_kwargs
        self.embed_kwargs = embed_kwargs
        self.normalize = normalize
        self.batch_size = batch_size

        self.nn = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar='tqdm'):
        self._validate_query_input(indices_unlabeled, n)

        if self.subsample_size > indices_unlabeled.shape[0]:
            return self.base_query_strategy.query(clf, dataset, indices_unlabeled, indices_labeled,
                                                  y, n=n)

        embeddings = clf.embed(dataset, pbar=pbar)
        if self.normalize:
            embeddings = normalize(embeddings, axis=1)

        indices_anchors = self._get_anchors(clf, embeddings, indices_labeled, y)
        indices_subset = self._get_subset_indices(embeddings,
                                                  indices_unlabeled,
                                                  indices_anchors)

        return self.base_query_strategy.query(clf, dataset, indices_subset, indices_labeled, y, n=n)

    def _get_anchors(self, clf, embeddings, indices_labeled, y):
        indices_anchors = []
        hist = _get_class_histogram(y, clf.num_classes)

        for c in range(clf.num_classes):
            num_samples = min(hist[c], self.num_anchors)
            if num_samples > 0:
                indices_labeled_per_class = indices_labeled[[i for i in range(y.shape[0]) if y[i] == c]]
                _, indices = init_kmeans_plusplus_safe(embeddings[indices_labeled_per_class],
                                                       num_samples,
                                                       x_squared_norms=np.linalg.norm(embeddings[indices_labeled_per_class], axis=1),
                                                       random_state=np.random.RandomState())
                indices_anchors.extend(indices_labeled_per_class[indices])

        return indices_anchors

    def _get_subset_indices(self, embeddings, indices_unlabeled, indices_achors):
        if self.nn is None:
            self.nn = self.initialize_index(embeddings, indices_unlabeled, self.hnsw_kwargs)
            self.indices_unlabeled = set(indices_unlabeled)
        else:
            recently_removed_elements = self.indices_unlabeled - set(indices_unlabeled)
            for el in recently_removed_elements:
                self.nn.mark_deleted(el)
            self.indices_unlabeled = set(indices_unlabeled)

        indices_nn, dists = self.nn.knn_query(embeddings[indices_achors], k=self.k)

        indices_nn = indices_nn.astype(int).flatten()
        similarity = 1 - dists.flatten()

        d = dict()
        for ind, sim in zip(indices_nn, similarity):
            d.setdefault(ind, []).append(sim)

        indices_nn = np.array(list(d.keys()))
        similarity = np.array([np.mean(v).item() for v in d.values()])

        if self.subsample_size >= indices_nn.shape[0]:
            return indices_nn

        indices_result = np.argpartition(-similarity, self.subsample_size)[:self.subsample_size]
        return indices_nn[indices_result]

    @staticmethod
    def initialize_index(embeddings, indices_unlabeled, hnsw_kwargs):
        import hnswlib

        space = hnsw_kwargs.get('space', 'l2')
        ef_construction = hnsw_kwargs.get('ef_construction', 200)
        m = hnsw_kwargs.get('M', 64)
        ef = hnsw_kwargs.get('ef', 200)

        index = hnswlib.Index(space=space, dim=embeddings.shape[1])
        index.init_index(max_elements=embeddings.shape[0], ef_construction=ef_construction, M=m)
        index.add_items(embeddings[indices_unlabeled], indices_unlabeled)
        index.set_ef(ef)

        return index

    def __str__(self):
        return f'AnchorSubsampling(base_query_strategy={str(self.base_query_strategy)}, ' + \
               f'num_anchors={self.num_anchors}, k={self.k}, hnsw_kwargs={self.hnsw_kwargs}, ' + \
               f'embed_kwargs={self.embed_kwargs}, normalize={self.normalize}, batch_size={self.batch_size})'
