import numpy.typing as npt

from typing import Protocol, TypeVar, Union
from scipy.sparse import csr_matrix


VIEW = TypeVar('VIEW', bound=object)


DATA = TypeVar('DATA', bound=object)


LABELS = TypeVar('LABELS', bound=object)


SKLEARN_DATA = Union[npt.NDArray, csr_matrix]


# TODO: use this for target labels or remove it
class HasDatasetProperties(Protocol):
    x: object
    y: object

    @property
    def multi_label(self) -> bool:
        pass
