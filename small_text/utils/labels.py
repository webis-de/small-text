import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

from small_text.base import LABEL_UNLABELED
from small_text.utils.annotations import deprecated


def get_num_labels(y):
    if y.shape[0] == 0:
        raise ValueError('Invalid labeling: Cannot contain 0 labels')

    if isinstance(y, csr_matrix):
        return np.max(y.indices) + 1
    else:
        return np.max(y) + 1


def remove_by_index(y, indices):
    if isinstance(y, csr_matrix):
        mask = np.ones(y.shape[0], dtype=bool)
        mask[indices] = False
        return y[mask, :]
    else:
        return np.delete(y, indices)


def get_ignored_labels_mask(y, ignored_label_value):
    if isinstance(y, csr_matrix):
        return np.array([(row.toarray() == ignored_label_value).any() for row in y])
    else:
        return y == np.array([ignored_label_value])


def concatenate(a, b):
    if isinstance(a, csr_matrix) and isinstance(b, csr_matrix):
        return vstack([a, b])
    else:
        return np.concatenate([a, b])


def csr_to_list(mat):
    return [mat.indices[tup[0]:tup[1]].tolist()
            for tup in list(zip(mat.indptr, mat.indptr[1:]))]


def list_to_csr(label_list, shape, dtype=np.int64):
    """Converts the given list of lists of labels into a sparse matrix.

    Parameter
    ---------
    label_list : list of list of int
        List of lists of labels.
    shape : tuple
        Target matrix shape.

    Returns
    -------
    result : scipy.sparse.csr_matrix
        A sparse matrix of the given input.
    """
    if np.all(np.array([len(item) for item in label_list]) == 0):
        return csr_matrix(shape, dtype=dtype)

    col_ind = [item if len(item) > 0 else np.empty(0, dtype=np.int64)
               for item in label_list]

    col_ind = np.concatenate(col_ind, dtype=np.int64)
    row_ind = np.concatenate([[i] * len(item)
                              for i, item in enumerate(label_list) if len(item)], dtype=np.int64)
    data = np.ones_like(col_ind, dtype=np.int64)

    return csr_matrix((data, (row_ind, col_ind)), shape=shape, dtype=dtype)


@deprecated(deprecated_in='1.1.0', to_be_removed_in='2.0.0')
def get_flattened_unique_labels(dataset):
    if dataset.is_multi_label:
        labels = np.unique(dataset.y.indices)
    else:
        labels = np.unique(dataset.y)
    return np.setdiff1d(labels, np.array([LABEL_UNLABELED]))
