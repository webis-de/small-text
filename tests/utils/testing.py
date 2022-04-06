import numpy as np

from scipy.sparse import csr_matrix

from numpy.testing import assert_array_equal, assert_raises


def assert_array_not_equal(x, y):
    assert_raises(AssertionError, assert_array_equal, x, y)


def assert_csr_matrix_equal(x, y):
    if x.shape != y.shape:
        raise AssertionError(f'Shape mismatch x: {x.shape} / y: {y.shape}')

    assert_array_equal(x.data, y.data)
    assert_array_equal(x.indices, y.indices)
    assert_array_equal(x.indptr, y.indptr)


def assert_csr_matrix_not_equal(x, y):
    assert_raises(AssertionError, assert_csr_matrix_equal, x, y)


def assert_labels_equal(x, y):
    if isinstance(x, csr_matrix) and isinstance(y, csr_matrix):
        assert_csr_matrix_equal(x, y)
    else:
        assert_array_equal(x, y)


def assert_list_of_tensors_equal(unittest_obj, input, other):
    import torch
    tensor_pairs = zip([item for item in input], [item for item in other])
    is_equal = [torch.equal(first, second)
                for first, second in tensor_pairs]
    unittest_obj.assertTrue(np.alltrue(is_equal))


def assert_list_of_tensors_not_equal(unittest_obj, input, other):
    assert_raises(AssertionError, assert_list_of_tensors_equal, unittest_obj, input, other)
