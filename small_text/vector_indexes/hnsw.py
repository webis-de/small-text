import numpy as np
from typing import Union, Optional

from small_text.base import check_optional_dependency
from small_text.vector_indexes.base import VectorIndex


class HNSWIndex(VectorIndex['hnswlib.Index']):
    """
    A vector index that relies on Hierarchical Navigable Small Worlds (HNSW).

    .. note ::
       This strategy requires the optional dependency `hnswlib`.

     .. seealso::
       GitHub repository of the underlying implementation.
           https://github.com/nmslib/hnswlib

       Details on setting the HNSW parameters.
           https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

    .. versionadded:: 2.0.0
    """

    def __init__(self, space='l2', ef_construction=200, ef=200, m=64, random_seed=100):
        """
        Parameters
        ----------
        space : {'l2', 'ip', 'cosine'}, default='l2'
            Type of vector space. The HNSW index uses the respective distance metric for
        ef_construction : int, default=200
            Parameter that trades index accuracy versus runtime. Higher values for `ef_construction`
            increase the indexing time.
        ef : int, default=200
            Parameter that trades index accuracy versus runtime. Higher values for `ef` increase the search time.
        m : int
            Number of links between vectors during construction. Data with higher intrinsic dimensionality requires
            higher values of `m`. Higher values of `m` increase the memory usage.
        random_seed : int
            Random seed that is used during initialization of the index.

        Note
        ----
        Check the `hnswlib GitHub repository <https://github.com/nmslib/hnswlib>`_ on details for the parameters.
        """
        check_optional_dependency('hnswlib')

        self.space = space
        self.ef_construction = ef_construction
        self.ef = ef
        self.m = m
        self.random_seed = random_seed

        self._index: Optional['hnswlib.Index'] = None

    @property
    def index(self) -> Optional['hnswlib.Index']:
        return self._index

    def build(self, vectors, ids=None):
        import hnswlib
        dim = vectors.shape[1]
        self._index = hnswlib.Index(space=self.space, dim=dim)
        self._index.init_index(max_elements=vectors.shape[0],
                               ef_construction=self.ef_construction,
                               M=self.m,
                               random_seed=self.random_seed)

        if ids is None:
            ids = np.arange(vectors.shape[0])

        self._index.add_items(vectors, ids)
        self._index.set_ef(self.ef)

    def remove(self, ids):
        for id_ in ids:
            self.index.mark_deleted(id_)

    def search(self, vectors, k: int = 10, return_distance: bool = False):
        try:
            indices, distances = self._index.knn_query(vectors, k=k)
        except RuntimeError as e:
            if 'Cannot return the results in a contiguous 2D array.' in str(e):
                raise ValueError(f'Searching the vector index failed. '
                                 f'Check if the given k={k} might exceeds the index size.')
            raise e

        if return_distance:
            return indices, distances
        else:
            return indices
