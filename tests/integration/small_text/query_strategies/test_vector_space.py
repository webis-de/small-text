import unittest

from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.query_strategies import ProbCover

from tests.unit.small_text.query_strategies.test_strategies import (
    SamplingStrategiesTests
)
from tests.utils.classification import SklearnClassifierWithRandomEmbeddings


class AdaptiveActiveLearningWithSingleLabelDataTest(unittest.TestCase, SamplingStrategiesTests):

    def _get_clf(self):
        return SklearnClassifierWithRandomEmbeddings(ConfidenceEnhancedLinearSVC(), 2)

    def _get_query_strategy(self):
        return ProbCover()
