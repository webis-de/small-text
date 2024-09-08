from abc import ABC, abstractmethod
from typing import Optional


class VectorIndex(ABC):
    """A structure that allows to index and search for vectors.

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
        """
        pass

    @abstractmethod
    def remove(self, vector_indices):
        pass

    @abstractmethod
    def search(self, vectors, k: int = 10, return_distance: bool = False):
        pass


class VectorIndexFactory(object):

    def __init__(self, vector_index_class, index_kwargs={}):
        self.vector_index_class = vector_index_class
        self.index_kwargs = index_kwargs

    def new(self) -> VectorIndex:
        return self.vector_index_class(**self.index_kwargs)
