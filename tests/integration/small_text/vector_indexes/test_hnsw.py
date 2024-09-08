import unittest

from small_text.vector_indexes.hnsw import HNSWIndex
from tests.integration.small_text.vector_indexes.test_base import VectorIndexesTest


class TestHNSWIndex(unittest.TestCase, VectorIndexesTest):

    def get_vector_index(self):
        return HNSWIndex()
