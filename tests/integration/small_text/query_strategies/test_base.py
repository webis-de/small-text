import unittest
import numpy as np

from numpy.testing import assert_array_almost_equal

from small_text.query_strategies.base import argselect


# Paranoid integration testing to verify that argselect is correct
class ArgselectIntegrationTest(unittest.TestCase):

    def test_argselect_maximum(self, n=5):

        for _ in range(1000):
            arr = np.random.randn(100)
            np.random.shuffle(arr)

            indices = argselect(arr, n)
            assert_array_almost_equal(np.sort(arr)[-5:], np.sort(arr[indices]))

    def test_argselect_no_tiebreak_maximum(self, n=5):

        for _ in range(1000):
            arr = np.random.randn(100)
            np.random.shuffle(arr)

            indices = argselect(arr, n, tiebreak=False)
            assert_array_almost_equal(np.sort(arr)[-5:], np.sort(arr[indices]))

    def test_argselect_minimum(self, n=5):
        np.random.seed(42)

        for _ in range(1000):
            arr = np.random.randint(10, size=100)
            np.random.shuffle(arr)

            indices = argselect(arr, n, maximum=False)
            assert_array_almost_equal(np.sort(arr)[:5], np.sort(arr[indices]))

    def test_argselect_no_tiebreak_minium(self, n=5):

        for _ in range(1000):
            arr = np.random.randn(100)
            np.random.shuffle(arr)

            indices = argselect(arr, n, tiebreak=False, maximum=False)
            assert_array_almost_equal(np.sort(arr)[:5], np.sort(arr[indices]))
