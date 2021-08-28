from numpy.testing import assert_array_equal, assert_raises


def assert_array_not_equal(x, y):
    assert_raises(AssertionError, assert_array_equal, x, y)
