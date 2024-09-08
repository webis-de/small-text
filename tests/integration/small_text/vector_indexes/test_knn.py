import unittest

from small_text.vector_indexes.knn import KNNIndex
from tests.integration.small_text.vector_indexes.test_base import VectorIndexesTest


class TestKNNIndex(unittest.TestCase, VectorIndexesTest):

    def get_vector_index(self):
        return KNNIndex()
