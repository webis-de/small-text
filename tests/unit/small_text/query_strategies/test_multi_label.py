import unittest
import numpy as np

from scipy.sparse import csr_matrix

from small_text.query_strategies.multi_label import (
    CategoryVectorInconsistencyAndRanking,
    _label_cardinality_inconsistency,
    LabelCardinalityInconsistency,
    _uncertainty_weighted_label_cardinality_inconsistency,
    AdaptiveActiveLearning
)
from small_text.utils.labels import list_to_csr


class LabelCardinalityFunctionTest(unittest.TestCase):

    def test_empty_labeled_set(self):
        with self.assertRaisesRegex(ValueError, 'Labeled pool labels must not be empty'):
            y_pred_proba_unlabeled = csr_matrix(
                np.array([
                    [0.0, 0.0, 0.0, 0.1],
                    [0.6, 0.0, 0.0, 0.1],
                    [0.6, 0.7, 0.0, 0.1],
                    [0.6, 0.7, 0.8, 0.1]
                ])
            )
            y_labeled = list_to_csr([], shape=(0, 3))
            _label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled)

    def test_label_cardinality_inconsistency_average_two_labels(self):
        y_pred_proba_unlabeled = csr_matrix(
            np.array([
                [0.0, 0.0, 0.0, 0.1],
                [0.6, 0.0, 0.0, 0.1],
                [0.6, 0.7, 0.0, 0.1],
                [0.6, 0.7, 0.8, 0.1]
            ])
        )
        y_labeled = list_to_csr([[0, 1], [1, 2]], shape=(2, 3))

        lci = _label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled)
        self.assertEqual((4,), lci.shape)
        self.assertTrue(lci[2] < lci[1] == lci[3] < lci[0])

    def test_label_cardinality_inconsistency_average_one_and_a_half_labels(self):
        y_pred_proba_unlabeled = csr_matrix(
            np.array([
                [0.0, 0.0, 0.0, 0.1],
                [0.6, 0.0, 0.0, 0.1],
                [0.6, 0.7, 0.0, 0.1],
                [0.6, 0.7, 0.8, 0.1]
            ])
        )
        y_labeled = list_to_csr([[0, 1], [1]], shape=(2, 3))

        lci = _label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled)
        self.assertEqual((4,), lci.shape)
        self.assertTrue(lci[1] == lci[2] < lci[0] == lci[0])

    def test_label_cardinality_inconsistency_empty_predictions(self):
        y_pred_unlabeled = list_to_csr([[], [], [], []], shape=(4, 3))
        y_labeled = list_to_csr([[0, 1], [1]], shape=(2, 3))

        lci = _label_cardinality_inconsistency(y_pred_unlabeled, y_labeled)
        self.assertEqual((4,), lci.shape)
        self.assertTrue(lci[0] == lci[1] == lci[2] == lci[3])

    def test_label_cardinality_inconsistency_empty_predictions_and_labels(self):
        y_pred_proba_unlabeled = csr_matrix(
            np.array([
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
        )
        y_labeled = list_to_csr([[], []], shape=(2, 3))

        lci = _label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled)
        self.assertEqual((4,), lci.shape)
        self.assertTrue(lci[0] == lci[1] == lci[2] == lci[3] == 0.0)


class LabelCardinalityInconsistencyTest(unittest.TestCase):

    def test_init(self):
        LabelCardinalityInconsistency()

    def test_init_with_nondefault_args(self):
        query_strategy = LabelCardinalityInconsistency(prediction_threshold=0.7)
        self.assertEqual('LabelCardinalityInconsistency(prediction_threshold=0.7)', str(query_strategy))

    def test_init_with_invalid_prediction_threshold(self):
        with self.assertRaisesRegex(ValueError, r'Prediction threshold must be in the interval \[0, 1\]'):
            AdaptiveActiveLearning(prediction_threshold=1.1)
        with self.assertRaisesRegex(ValueError, r'Prediction threshold must be in the interval \[0, 1\]'):
            AdaptiveActiveLearning(prediction_threshold=-1.1)

    def test_str(self):
        query_strategy = LabelCardinalityInconsistency()
        self.assertEqual('LabelCardinalityInconsistency(prediction_threshold=0.5)', str(query_strategy))


class UncertaintyWeightedLabelCardinalityFunctionTest(unittest.TestCase):

    def test_empty_labeled_set(self):
        with self.assertRaisesRegex(ValueError, 'Labeled pool labels must not be empty'):
            y_pred_proba_unlabeled = np.array([
                [0.0, 0.5, 0.55, 0.9],
                [0.8, 0.3, 0.8, 0.2],
                [0.0, 0.99, 0.6, 0.1],
                [0.2, 0.2, 0.3, 0.4]
            ])
            y_labeled = list_to_csr([], shape=(0, 4))
            _uncertainty_weighted_label_cardinality_inconsistency(y_pred_proba_unlabeled,
                                                                 y_labeled)

    def test_uncertainty_weighted_label_cardinality_inconsistency_average_two_labels(self):
        y_pred_proba_unlabeled = np.array([
            [0.0, 0.5, 0.55, 0.9],
            [0.8, 0.3, 0.8, 0.2],
            [0.0, 0.99, 0.6, 0.1],
            [0.2, 0.2, 0.6, 0.6]
        ])
        y_labeled = list_to_csr([[0, 1], [1, 2]], shape=(2, 4))

        uwlci = _uncertainty_weighted_label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled)
        self.assertEqual((4,), uwlci.shape)
        self.assertTrue(uwlci[2] < uwlci[0] < uwlci[1] == uwlci[3])

    def test_label_cardinality_inconsistency_average_one_and_a_half_labels(self):
        y_pred_proba_unlabeled = np.array([
            [0.0, 0.5, 0.55, 0.9],
            [0.8, 0.3, 0.8, 0.2],
            [0.0, 0.99, 0.6, 0.1],
            [0.2, 0.2, 0.3, 0.4]
        ])
        y_labeled = list_to_csr([[0, 1], [1]], shape=(2, 4))

        uwlci = _uncertainty_weighted_label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled)
        self.assertEqual((4,), uwlci.shape)
        self.assertTrue(uwlci[2] < uwlci[0] < uwlci[3] < uwlci[1])

    def test_uncertainty_weighted_label_cardinality_empty_predictions(self):
        y_pred_proba_unlabeled = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
        y_labeled = list_to_csr([[0, 1], [1, 2]], shape=(2, 4))

        uwlci = _uncertainty_weighted_label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled)
        self.assertEqual((4,), uwlci.shape)
        self.assertTrue(uwlci[0] == uwlci[1] == uwlci[2] == uwlci[3])


class AdaptiveActiveLearningTest(unittest.TestCase):

    def test_init(self):
        AdaptiveActiveLearning()

    def test_init_with_nondefault_args(self):
        query_strategy = AdaptiveActiveLearning(uncertainty_weight=0.7, prediction_threshold=0.7)
        self.assertEqual('AdaptiveActiveLearning(uncertainty_weight=0.7, prediction_threshold=0.7)',
                         str(query_strategy))

    def test_init_with_invalid_uncertainty_weight(self):
        with self.assertRaisesRegex(ValueError, r'Uncertainty weight must be in the interval \[0, 1\]'):
            AdaptiveActiveLearning(uncertainty_weight=1.1)
        with self.assertRaisesRegex(ValueError, r'Uncertainty weight must be in the interval \[0, 1\]'):
            AdaptiveActiveLearning(uncertainty_weight=-1.1)

    def test_init_with_invalid_prediction_threshold(self):
        with self.assertRaisesRegex(ValueError, r'Prediction threshold must be in the interval \[0, 1\]'):
            AdaptiveActiveLearning(prediction_threshold=1.1)
        with self.assertRaisesRegex(ValueError, r'Prediction threshold must be in the interval \[0, 1\]'):
            AdaptiveActiveLearning(prediction_threshold=-1.1)

    def test_str(self):
        query_strategy = AdaptiveActiveLearning()
        self.assertEqual('AdaptiveActiveLearning(uncertainty_weight=0.5, prediction_threshold=0.5)',
                         str(query_strategy))


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
