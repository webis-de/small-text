import numpy as np

from typing import Union

from sklearn.neighbors import NearestNeighbors
from small_text.vector_indexes.base import VectorIndex


class KNNIndex(VectorIndex):
    """
    A vector index that relies on unsupervised learning of K-Nearest Neighbors.

     .. seealso::
       Scikit-learn documentation of the underlying implementation.
           https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

    .. versionadded:: 2.0.0
    """

    DEFAULT_NUM_NEIGHBORS = 10

    def __init__(self, algorithm='auto', radius=1.0, leaf_size=30, metric='minkowski', p=2,
                 metric_params=None, n_jobs=None):
        """
        Parameters
        ----------
        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
            Algorithm used for the nearest neighbor computation.
        radius : float, default=1.0
            Ball of size `radius`
        leaf_size : int, default=30
            Leaf size for 'ball_tree' and 'kd_tree' algorithms.
        metric : str or func, default='minkowski'
            Metric or metric function for the nearest neighbor distance.
        p : float, default=2
            Parameter p for the Minkowski distance. The default `2` is equivalent to the euclidean distance.
        metric_params : dict, default=None
            Additional params for the metric function.
        n_jobs : int, default=None
            Number of jobs for nearest neighbor search.

        .. seealso::
           See the scikit-learn documentation for more details on the parameters.
               https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

        """
        self.algorithm = algorithm
        self.radius = radius
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self._index = None
        self._ids_to_indices = None
        self._indices_to_ids = None
        self._indices_deleted = set()

    @property
    def index(self) -> Union[None, object]:
        return self._index

    def build(self, vectors, ids=None):
        self._index = NearestNeighbors(n_neighbors=self.DEFAULT_NUM_NEIGHBORS,
                                       algorithm=self.algorithm,
                                       metric=self.metric,
                                       p=self.p,
                                       metric_params=self.metric_params,
                                       n_jobs=self.n_jobs)
        self._index.fit(vectors)

        if ids is None:
            ids = np.arange(vectors.shape[0])

        self._ids_to_indices = {id_: idx for idx, id_ in enumerate(ids)}
        self._indices_to_ids = {idx: id_ for idx, id_ in enumerate(ids)}

        self._indices_deleted = set()

    def remove(self, ids):
        for id_ in ids.tolist():
            self._indices_deleted.add(self._ids_to_indices[id_])

    def search(self, vectors, k: int = 10, return_distance: bool = False):
        try:
            result = self._index.kneighbors(X=vectors,
                                            n_neighbors=k+len(self._indices_deleted),
                                            return_distance=return_distance)
        except ValueError as e:
            if 'Expected n_neighbors <= n_samples' in str(e):
                raise ValueError(f'Searching the vector index failed. '
                                 f'Check if the given k={k} might exceeds the index size.')
            raise e

        if return_distance:
            distances = np.array([[idx for idx in indices if idx.item() not in self._indices_deleted][:k]
                                   for indices in result[0]])
            indices = np.array([
                [self._indices_to_ids[idx] for idx in indices if idx.item() not in self._indices_deleted][:k]
                for indices in result[1]
            ])
            return indices, distances
        else:
            return np.array([
                [self._indices_to_ids[idx] for idx in indices if idx.item() not in self._indices_deleted][:k]
                for indices in result
            ])
