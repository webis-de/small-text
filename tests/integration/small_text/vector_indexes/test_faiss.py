import faiss

import unittest

from small_text.vector_indexes.faiss import FaissIndex
from tests.integration.small_text.vector_indexes.test_base import VectorIndexesTest
from tests.utils.pytest import mark_optional_dependency_test

@mark_optional_dependency_test('faiss')
class TestFaissIndex(unittest.TestCase, VectorIndexesTest):

    def get_vector_index(self):
        return FaissIndex()
