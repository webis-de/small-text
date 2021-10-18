import numpy as np
from numpy.testing import assert_array_equal, assert_raises


def assert_array_not_equal(x, y):
    assert_raises(AssertionError, assert_array_equal, x, y)


def assert_list_of_tensors_equal(unittest_obj, input, other):
    import torch
    tensor_pairs = zip([item for item in input], [item for item in other])
    is_equal = [torch.equal(first, second)
                for first, second in tensor_pairs]
    unittest_obj.assertTrue(np.alltrue(is_equal))
