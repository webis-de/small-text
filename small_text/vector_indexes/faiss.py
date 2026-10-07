import numpy as np

from typing import Optional

from small_text.vector_indexes.base import VectorIndex
from small_text.exceptions import MissingOptionalDependencyError

def _check_faiss_dependency():
    """Check whether FAISS is installed and importable."""
    try:
        import faiss
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "FAISS is required. Install faiss-cpu or faiss-gpu."
        ) from exc

    return faiss

class FaissIndex(VectorIndex['faiss.Index']):
    """
    A vector index that relies on FAISS (Facebook AI Similarity Search).

    .. note::
       This strategy requires FAISS. CPU execution is supported by
       the CPU package; GPU execution requires a GPU-enabled installation.

    .. seealso::
       GitHub repository of the underlying implementation.
           https://github.com/facebookresearch/faiss

    .. versionadded:: 2.0.0
    """

    def __init__(self, device='cpu', gpu_id=0):
        _check_faiss_dependency()

        if device not in ('cpu', 'cuda'):
            raise ValueError("device must be either 'cpu' or 'cuda'")

        self.device = device
        self.gpu_id = gpu_id
        self._index: Optional['faiss.Index'] = None
        self._resources = None

        if self.device == 'cuda':
            self._check_gpu_support()

    def _check_gpu_support(self):
        import faiss

        required_attributes = (
            'StandardGpuResources',
            'index_cpu_to_gpu',
            'get_num_gpus',
        )

        if not all(hasattr(faiss, attr) for attr in required_attributes):
            raise RuntimeError(
                'This FAISS installation does not support GPU operations.'
            )

        if (
            not isinstance(self.gpu_id, int)
            or isinstance(self.gpu_id, bool)
            or self.gpu_id < 0
        ):
            raise ValueError('gpu_id must be a non-negative integer.')

        if self.gpu_id >= faiss.get_num_gpus():
            raise RuntimeError(
                f'CUDA GPU {self.gpu_id} is not available.'
            )

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

        cpu_index = faiss.IndexFlatL2(dimension)
        cpu_index = faiss.IndexIDMap(cpu_index)
        cpu_index.add_with_ids(vectors, ids)

        if self.device == 'cuda':
            self._resources = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(
                self._resources,
                self.gpu_id,
                cpu_index,
            )
        else:
            self._index = cpu_index

    def search(self, vectors, k: int = 10, return_distance: bool = False):
        vectors = np.asarray(vectors, dtype=np.float32)

        if self.device == 'cuda' and self._index is None:
            raise ValueError('The vector index has not been built.')

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
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k <= 0:
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
        import faiss

        if self.index is None:
            raise ValueError('The vector index has not been built.')

        ids = np.asarray(ids, dtype=np.int64)

        if ids.ndim != 1:
            raise ValueError('IDs must be a one-dimensional array.')

        if self.device == 'cuda':
            cpu_index = faiss.index_gpu_to_cpu(self._index)
            cpu_index.remove_ids(ids)

            self._index = faiss.index_cpu_to_gpu(
                self._resources,
                self.gpu_id,
                cpu_index,
            )
        else:
            self._index.remove_ids(ids)
