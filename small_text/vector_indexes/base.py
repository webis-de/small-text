from abc import ABCMeta, abstractmethod
from typing import Generic, Optional

from small_text.vector_indexes._typing import INDEX_TYPE


class VectorIndex(Generic[INDEX_TYPE], metaclass=ABCMeta):
    """Abstract class for a structure that allows to index and search for vectors.

    .. versionadded:: 2.0.0
    """

    @property
    @abstractmethod
    def index(self) -> Optional[INDEX_TYPE]:
        """
        Returns the underlying index implementation.

        Returns
        -------
        index : object or None
            The underlying index if an index has been built, otherwise None.
        """
        return

    @abstractmethod
    def build(self, vectors, ids=None):
        """
        Constructs an index from the given vectors. Each vector is identified by an id. If no ids are passed,
        each vector gets assigned an ascending id starting at zero.

        Parameters
        ----------
        vectors : np.ndarray[np.float32]
            A 2d matrix of vectors in the shape (num_vectors, num_dimensions).
        ids : np.ndarray[int] or None, default=None
            An array of ids where each item corresponds to the respective row in the `vectors` argument.
        """
        pass

    @abstractmethod
    def remove(self, ids):
        """
        Removes the given vectors (identified by their ids) from the vector index.
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
        ids : np.ndarray[int]
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
