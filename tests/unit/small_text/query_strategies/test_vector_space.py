import unittest
import numpy as np

from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.query_strategies.vector_space import ProbCover

from tests.utils.datasets import random_sklearn_dataset


class BALDTest(unittest.TestCase):

    def _get_clf(self):
        return ConfidenceEnhancedLinearSVC()

    def _get_query_strategy(self):
        return ProbCover()

    def test_str(self):
        strategy = self._get_query_strategy()
        expected_str = 'ProbCover(vector_index_factory=VectorIndexFactory, ball_radius=0.1, k=100)'
        self.assertEqual(expected_str, str(strategy))
