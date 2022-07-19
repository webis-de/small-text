import unittest

from small_text.query_strategies import CategoryVectorInconsistencyAndRanking


class CategoryVectorInconsistencyAndRankingTest(unittest.TestCase):

    def test_init(self):
        query_strategy = CategoryVectorInconsistencyAndRanking()

        self.assertEqual(2048, query_strategy.batch_size)
        self.assertEqual(0.5, query_strategy.prediction_threshold)
        self.assertEqual(1e-8, query_strategy.epsilon)

    def test_init_with_non_default_arguments(self):
        batch_size = 4096
        prediction_threshold = 0.45
        epsilon = 1e-10

        query_strategy = CategoryVectorInconsistencyAndRanking(
            batch_size=batch_size,
            prediction_threshold=prediction_threshold,
            epsilon=epsilon
        )

        self.assertEqual(batch_size, query_strategy.batch_size)
        self.assertEqual(prediction_threshold, query_strategy.prediction_threshold)
        self.assertEqual(epsilon, query_strategy.epsilon)

    def test_entropy(self):
        query_strategy = CategoryVectorInconsistencyAndRanking()
        self.assertEqual(0, query_strategy._entropy(0, 10))

    def test_str(self):
        query_strategy = CategoryVectorInconsistencyAndRanking()
        self.assertEqual(
            'CategoryVectorInconsistencyAndRanking('
            'batch_size=2048, prediction_threshold=0.5, epsilon=1e-08)',
            str(query_strategy))
