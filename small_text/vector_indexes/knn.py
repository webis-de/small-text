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

    def __init__(self, index_kwargs={}):
        self.index_kwargs = index_kwargs
        self._index = None
        self._indices_deleted = set()

    @property
    def index(self) -> Union[None, object]:
        return self._index

    def build(self, vectors):
        self._index = NearestNeighbors(n_neighbors=self.DEFAULT_NUM_NEIGHBORS, **self.index_kwargs)
        self._index.fit(vectors)
        self._indices_deleted = set()

    def remove(self, vector_indices):
        for idx in vector_indices.tolist():
            self._indices_deleted.add(idx)

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
            indices = np.array([[idx for idx in indices if idx.item() not in self._indices_deleted]
                                for indices in result[0]])
            distances = np.array([[idx for idx in indices if idx.item() not in self._indices_deleted]
                                   for indices in result[1]])
            return indices, distances
        else:
            return np.array([[idx for idx in indices if idx.item() not in self._indices_deleted]
                            for indices in result])
