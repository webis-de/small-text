import unittest
import numpy as np

from active_learning.utils.data import list_length


class DataUtilsTest(unittest.TestCase):

    def test_list_length(self):
        self.assertEqual(10, list_length(list(range(10))))
        self.assertEqual(10, list_length(np.random.rand(10, 2)))
