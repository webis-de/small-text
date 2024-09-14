from abc import ABC, abstractmethod
from typing import Optional


class VectorIndex(ABC):
    """Abstract class for a structure that allows to index and search for vectors.

    .. versionadded:: 2.0.0
    """

    @property
    @abstractmethod
    def index(self) -> Optional[object]:
        """
        Returns the underlying index implementation.

        Returns
        -------
        index : object or None
            The underlying index if an index has been built, otherwise None.
        """
        return

    @abstractmethod
    def build(self, vectors):
        """
        Constructs an index from the given vectors.

        The order of `vectors` is later used to address them in the `remove()` operation.

        Parameters
        ----------
        vectors : np.ndarray[np.float32]
            A 2d matrix of vectors in the shape (num_vectors, num_dimensions).
        """
        pass

    @abstractmethod
    def remove(self, vector_indices):
        """
        Removes the given vectors (identified by the numeric indices) from the vector index. The indices
         `vector_indices` correspond to the rows numbers in the matrix of vectors that has been passed to `build()`.
        """
        pass

    @abstractmethod
    def search(self, vectors, k: int = 10, return_distance: bool = False):
        """
        For each of the given vectors, retrieve and return the `k` most similar vectors from the index.

        Parameters
        ----------
        vectors : np.ndarray[np.float32]
            A 2d matrix of vectors in the shape (num_vectors, num_dimensions).
        k : int, k=10
            Specified the number of similar vectors that are returned for each input vector.
        return_distance : bool, default=False
            Toggles if the distances should be returned in addition to the vector indices.

        Returns
        -------
        indices : np.ndarray[int]
            A 2d matrix of vectors in the shape (num_vectors, k) which holds `k` indices per row. The indices
            refer to the vectors on which the index has been built, i.e.
        distances : np.ndarray[np.float32]
            A 2d matrix of vectors in the shape (num_vectors, k) which holds normalized distances between
            `0.0` (most similar) and `1.0` (most dissimilar). Distances are only returned if `return_distance` is `True`.
        """
        pass


class VectorIndexFactory(object):

    def __init__(self, vector_index_class, index_kwargs={}):
        self.vector_index_class = vector_index_class
        self.index_kwargs = index_kwargs

    def new(self) -> VectorIndex:
        return self.vector_index_class(**self.index_kwargs)
