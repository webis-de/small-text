import unittest

import numpy as np
from numpy.testing import assert_array_equal

from small_text.stopping_criteria.change import ClassificationChange
from small_text.utils.labels import list_to_csr

from tests.utils.testing import assert_csr_matrix_equal


class ClassificationChangeTest(unittest.TestCase):

    def test_init(self):
        stopping_criterion = ClassificationChange(2)

        self.assertEqual(2, stopping_criterion.num_classes)
        self.assertEqual(0, stopping_criterion.threshold)

    def test_init_with_kwargs(self):

        num_classes = 2
        threshold = 0.1

        stopping_criterion = ClassificationChange(num_classes, threshold=threshold)

        self.assertEqual(num_classes, stopping_criterion.num_classes)
        self.assertEqual(threshold, stopping_criterion.threshold)

    def test_init_with_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, 'Threshold must be between 0 and 1 inclusive'):
            ClassificationChange(2, threshold=-1)
        with self.assertRaisesRegex(ValueError, 'Threshold must be between 0 and 1 inclusive'):
            ClassificationChange(2, threshold=1.01)


class ClassificationChangeSingleLabelStopTest(unittest.TestCase):

    def test_first_stop_call(self):
        stopping_criterion = ClassificationChange(2)

        predictions = np.array([0, 1, 1, 0])

        stop = stopping_criterion.stop(predictions=predictions)
        self.assertFalse(stop)

        assert_array_equal(predictions, stopping_criterion.last_predictions)

    def test_second_stop_call(self):
        stopping_criterion = ClassificationChange(2)

        first_predictions = np.array([0, 1, 1, 0])
        second_predictions = np.array([0, 1, 1, 1])

        stopping_criterion.stop(predictions=first_predictions)
        stop = stopping_criterion.stop(predictions=second_predictions)
        self.assertFalse(stop)

        assert_array_equal(second_predictions, stopping_criterion.last_predictions)

    def test_stop(self):
        stopping_criterion = ClassificationChange(2, threshold=0.25)

        first_predictions = np.array([0, 1, 1, 0])
        second_predictions = np.array([1, 0, 1, 0])
        third_predictions = np.array([0, 1, 0, 1])
        fourth_predictions = np.array([0, 1, 1, 1])

        self.assertFalse(stopping_criterion.stop(predictions=first_predictions))
        self.assertFalse(stopping_criterion.stop(predictions=second_predictions))
        self.assertFalse(stopping_criterion.stop(predictions=third_predictions))
        self.assertTrue(stopping_criterion.stop(predictions=fourth_predictions))

        assert_array_equal(fourth_predictions, stopping_criterion.last_predictions)

    def test_stop_with_predictions_none(self):
        stopping_criterion = ClassificationChange(2)

        with self.assertRaises(ValueError):
            stopping_criterion.stop()

        with self.assertRaises(ValueError):
            stopping_criterion.stop(predictions=np.array([None, None, None]))

    def test_stop_with_prediction_size_changing(self):
        stopping_criterion = ClassificationChange(2)

        with self.assertRaises(ValueError):
            stopping_criterion.stop(predictions=np.array([0, 1, 0]))
            stopping_criterion.stop(predictions=np.array([0, 1, 0, 1]))


class ClassificationChangeMultiLabelStopTest(unittest.TestCase):

    def test_first_stop_call(self):
        stopping_criterion = ClassificationChange(2)

        predictions = list_to_csr([[], [0, 1], [1], [0]], shape=(4, 2))

        stop = stopping_criterion.stop(predictions=predictions)
        self.assertFalse(stop)

        assert_csr_matrix_equal(predictions, stopping_criterion.last_predictions)

    def test_second_stop_call(self):
        stopping_criterion = ClassificationChange(2)

        first_predictions = list_to_csr([[], [0, 1], [1], [0]], shape=(4, 2))
        second_predictions = list_to_csr([[], [0, 1], [1], [1]], shape=(4, 2))

        stopping_criterion.stop(predictions=first_predictions)
        stop = stopping_criterion.stop(predictions=second_predictions)
        self.assertFalse(stop)

        assert_csr_matrix_equal(second_predictions, stopping_criterion.last_predictions)

    def test_stop(self):
        stopping_criterion = ClassificationChange(2, threshold=0.25)

        first_predictions = list_to_csr([[], [0, 1], [1], [0]], shape=(4, 2))
        second_predictions = list_to_csr([[1], [0, 1], [0, 1], [0]], shape=(4, 2))
        third_predictions = list_to_csr([[1], [1], [0, 1], []], shape=(4, 2))
        fourth_predictions = list_to_csr([[1], [], [0, 1], []], shape=(4, 2))

        self.assertFalse(stopping_criterion.stop(predictions=first_predictions))
        self.assertFalse(stopping_criterion.stop(predictions=second_predictions))
        self.assertFalse(stopping_criterion.stop(predictions=third_predictions))
        self.assertTrue(stopping_criterion.stop(predictions=fourth_predictions))

        assert_csr_matrix_equal(fourth_predictions, stopping_criterion.last_predictions)

    def test_stop_with_predictions_empty(self):
        stopping_criterion = ClassificationChange(2)

        with self.assertRaises(ValueError):
            stopping_criterion.stop()

        self.assertFalse(stopping_criterion.stop(predictions=list_to_csr([[], [], [], []], shape=(4, 2))))

    def test_stop_with_prediction_size_changing(self):
        stopping_criterion = ClassificationChange(2)

        with self.assertRaises(ValueError):
            stopping_criterion.stop(predictions=list_to_csr([[], [1], [0]], shape=(3, 2)))
            stopping_criterion.stop(predictions=list_to_csr([[], [0, 1], [1], [0]], shape=(4, 2)))
