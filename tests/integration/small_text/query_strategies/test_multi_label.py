import unittest

from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.query_strategies import CategoryVectorInconsistencyAndRanking

from tests.unit.small_text.query_strategies.test_strategies import (
    SamplingStrategiesTests,
    SklearnClassifierWithRandomEmbeddings
)


class CategoryVectorInconsistencyAndRankingWithSingleLabelDataTest(unittest.TestCase,
                                                                   SamplingStrategiesTests):

    def _get_clf(self):
        return SklearnClassifierWithRandomEmbeddings(ConfidenceEnhancedLinearSVC(), 2)

    def _get_query_strategy(self):
        return CategoryVectorInconsistencyAndRanking()

    def test_simple_query(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires '
                                    'classification_type=multi-label'):
            super().test_simple_query()

    def test_default_query(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires '
                                    'classification_type=multi-label'):
            super().test_default_query()

    def test_query_takes_remaining_pool(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires '
                                    'classification_type=multi-label'):
            super().test_query_takes_remaining_pool()

    def test_query_exhausts_pool(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'Invalid configuration: This query strategy requires '
                                    'classification_type=multi-label'):
            super().test_query_exhausts_pool()


class CategoryVectorInconsistencyAndRankingTest(unittest.TestCase, SamplingStrategiesTests):

    def _get_clf(self):
        return SklearnClassifierWithRandomEmbeddings(ConfidenceEnhancedLinearSVC(), 3,
                                                     multi_label=True)

    def _get_query_strategy(self):
        return CategoryVectorInconsistencyAndRanking()

    def _is_multi_label(self):
        return True
