import unittest

from small_text.vector_indexes.hnsw import HNSWIndex
from tests.integration.small_text.vector_indexes.test_base import VectorIndexesTest
from tests.utils.pytest import mark_optional_dependency_test


@mark_optional_dependency_test('hnswlib')
class TestHNSWIndex(unittest.TestCase, VectorIndexesTest):

    def get_vector_index(self):
        return HNSWIndex()
