import unittest
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from small_text.stopping_criteria.base import DeltaFScore


class DeltaFScoreTest(unittest.TestCase):

    def test_init_default(self):
        stopping_criterion = DeltaFScore(2)

        self.assertEqual(2, stopping_criterion.num_classes)
        self.assertEqual(3, stopping_criterion.window_size)
        self.assertEqual(0.05, stopping_criterion.threshold)
        self.assertIsNone(stopping_criterion.last_predictions)
        self.assertEqual([], stopping_criterion.delta_history)

    def test_init_with_kwargs(self):

        window_size = 5
        threshold = 0.01

        stopping_criterion = DeltaFScore(2, window_size=window_size, threshold=threshold)

        self.assertEqual(2, stopping_criterion.num_classes)
        self.assertEqual(5, stopping_criterion.window_size)
        self.assertEqual(0.01, stopping_criterion.threshold)
        self.assertIsNone(stopping_criterion.last_predictions)
        self.assertEqual([], stopping_criterion.delta_history)

    def test_init_multiclass(self):
        with self.assertRaises(ValueError):
            DeltaFScore(3)

    def test_first_stop_call(self):
        stopping_criterion = DeltaFScore(2)

        predictions = np.array([0, 1, 1, 0])

        stop = stopping_criterion.stop(predictions=predictions)
        self.assertFalse(stop)

        self.assertEqual([], stopping_criterion.delta_history)
        assert_array_equal(predictions, stopping_criterion.last_predictions)

    def test_second_stop_call(self):
        stopping_criterion = DeltaFScore(2)

        first_predictions = np.array([0, 1, 1, 0])
        second_predictions = np.array([0, 1, 1, 1])

        stopping_criterion.stop(predictions=first_predictions)
        stop = stopping_criterion.stop(predictions=second_predictions)
        self.assertFalse(stop)

        assert_array_almost_equal([0.142857], stopping_criterion.delta_history)
        assert_array_equal(second_predictions, stopping_criterion.last_predictions)

    def test_stop(self):
        stopping_criterion = DeltaFScore(2, threshold=0.1)

        first_predictions = np.array([0, 1, 1, 0, 1, 1, 0])
        second_predictions = np.array([0, 1, 1, 1, 1, 1, 0])
        third_predictions = np.array([1, 1, 1, 1, 1, 1, 0])
        fourth_predictions = np.array([1, 1, 1, 0, 1, 1, 0])

        stopping_criterion.stop(predictions=first_predictions)

        self.assertFalse(stopping_criterion.stop(predictions=second_predictions))
        assert_array_almost_equal([0.076923], stopping_criterion.delta_history)
        assert_array_equal(second_predictions, stopping_criterion.last_predictions)

        self.assertFalse(stopping_criterion.stop(predictions=third_predictions))
        assert_array_almost_equal([0.076923, 0.076923], stopping_criterion.delta_history)
        assert_array_equal(third_predictions, stopping_criterion.last_predictions)

        self.assertTrue(stopping_criterion.stop(predictions=fourth_predictions))
        assert_array_almost_equal([0.076923, 0.076923, 0.076923], stopping_criterion.delta_history)
        assert_array_equal(fourth_predictions, stopping_criterion.last_predictions)

    def test_stop_with_predictions_none(self):
        stopping_criterion = DeltaFScore(2)

        with self.assertRaises(ValueError):
            stopping_criterion.stop()

        with self.assertRaises(ValueError):
            stopping_criterion.stop(predictions=np.array([None, None, None]))

    def test_stop_with_prediction_size_changing(self):
        stopping_criterion = DeltaFScore(2)

        with self.assertRaises(ValueError):
            stopping_criterion.stop(predictions=np.array([0, 1, 0]))
            stopping_criterion.stop(predictions=np.array([0, 1, 0, 1]))
