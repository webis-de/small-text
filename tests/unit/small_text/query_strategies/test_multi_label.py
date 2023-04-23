import unittest

from small_text.query_strategies.multi_label import (
    CategoryVectorInconsistencyAndRanking,
    label_cardinality_inconsistency,
    LabelCardinalityInconsistency
)
from small_text.utils.labels import list_to_csr


class LabelCardinalityFunctionTest(unittest.TestCase):

    def test_empty_labeled_set(self):
        with self.assertRaisesRegex(ValueError, 'Labeled pool labels must not be empty'):
            y_pred_unlabeled = list_to_csr([[], [0], [0, 1], [0, 1, 2]], shape=(4, 3))
            y_labeled = list_to_csr([], shape=(0, 3))
            label_cardinality_inconsistency(y_pred_unlabeled, y_labeled)

    def test_label_cardinality_inconsistency_average_two_labels(self):
        y_pred_unlabeled = list_to_csr([[], [0], [0, 1], [0, 1, 2]], shape=(4, 3))
        y_labeled = list_to_csr([[0, 1], [1, 2]], shape=(2, 3))

        lci = label_cardinality_inconsistency(y_pred_unlabeled, y_labeled)
        self.assertEqual((4,), lci.shape)
        self.assertTrue(lci[2] < lci[1] == lci[3] < lci[0])

    def test_label_cardinality_inconsistency_average_one_and_a_half_labels(self):
        y_pred_unlabeled = list_to_csr([[], [0], [0, 1], [0, 1, 2]], shape=(4, 3))
        y_labeled = list_to_csr([[0, 1], [1]], shape=(2, 3))

        lci = label_cardinality_inconsistency(y_pred_unlabeled, y_labeled)
        self.assertEqual((4,), lci.shape)
        self.assertTrue(lci[1] == lci[2] < lci[0] == lci[0])


class LabelCardinalityInconsistencygTest(unittest.TestCase):

    def test_init(self):
        LabelCardinalityInconsistency()

    def test_str(self):
        query_strategy = LabelCardinalityInconsistency()
        self.assertEqual('LabelCardinalityInconsistency()', str(query_strategy))


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
