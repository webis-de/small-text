import numpy as np

from typing import Optional

from small_text.base import check_optional_dependency
from small_text.vector_indexes.base import VectorIndex

class FaissIndex(VectorIndex['faiss.Index']):
    """
    A vector index that relies on FAISS (Facebook AI Similarity Search).

    .. note ::
       This strategy requires the optional dependency `faiss-cpu`.

    .. seealso::
       GitHub repository of the underlying implementation.
           https://github.com/facebookresearch/faiss

    .. versionadded:: 2.0.0
    """

    def __init__(self):
        check_optional_dependency('faiss')
        self._index: Optional['faiss.Index'] = None

    @property
    def index(self) -> Optional['faiss.Index']:
        return self._index

    def build(self, vectors, ids=None):
        """
        Create the FAISS index from a collection of vectors.

        Parameters
        ----------
        vectors : numpy.ndarray of shape (n_vectors, n_features)
            The vectors that are to be added to the index.
        ids : numpy.ndarray of shape (n_vectors,), optional
            The integer IDs to associate with the vectors. If `None`, sequential IDs
            starting from 0 will be assigned automatically.

        Raises
        ------
        ValueError
            If the vectors are empty, not two-dimensional, or have zero features,
            or if the IDs do not match the number of vectors or are not one-dimensional.
        """
        import faiss     # Optional dependency; a global import may raise ImportError

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError('Vectors must be a non-empty 2D array.')

        dimension = vectors.shape[1]
        if dimension == 0:
            raise ValueError('Vectors must have at least one feature.')

        if ids is None:
            ids = np.arange(vectors.shape[0], dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)

        if ids.ndim != 1 or len(ids) != vectors.shape[0]:
            raise ValueError('The number of IDs must match the number of vectors.')

        self._index = faiss.IndexFlatL2(dimension)

        self._index = faiss.IndexIDMap(self._index)
        self._index.add_with_ids(vectors, ids)

    def search(self, vectors, k: int = 10, return_distance: bool = False):
        vectors = np.asarray(vectors, dtype=np.float32)

        if self.index is None:
            raise ValueError('The vector index has not been built.')
        if (
            vectors.ndim != 2
            or vectors.shape[0] == 0
            or vectors.shape[1] != self.index.d
        ):
            raise ValueError(
                'Query vectors must be a non-empty 2D array with the same dimension as indexed vectors.'
            )
        if k <= 0:
            raise ValueError('k must be a positive integer.')
        if k > self.index.ntotal:
            raise ValueError(
                f'Searching the vector index failed. '
                f'Check if the given k={k} might exceed the index size.'
            )

        distances, indices = self._index.search(vectors, k)

        if return_distance is True:
            return indices, distances
        else:
            return indices

    def remove(self, ids):
        self.index.remove_ids(np.asarray(ids, dtype=np.int64))