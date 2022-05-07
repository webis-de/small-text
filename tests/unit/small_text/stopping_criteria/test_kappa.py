import unittest

import numpy as np
from numpy.testing import assert_array_equal

from small_text.stopping_criteria.kappa import KappaAverage


class KappaAverageTest(unittest.TestCase):

    def test_init(self):
        stopping_criterion = KappaAverage(2)

        self.assertEqual([], stopping_criterion.kappa_history)
        self.assertEqual(3, stopping_criterion.window_size)
        self.assertEqual(0.99, stopping_criterion.kappa)
        self.assertEqual(2, stopping_criterion.num_classes)

    def test_init_with_kwargs(self):

        window_size = 5
        kappa = 0.95
        num_classes = 5

        stopping_criterion = KappaAverage(num_classes, kappa=kappa, window_size=window_size)

        self.assertEqual([], stopping_criterion.kappa_history)
        self.assertEqual(window_size, stopping_criterion.window_size)
        self.assertEqual(kappa, stopping_criterion.kappa)
        self.assertEqual(num_classes, stopping_criterion.num_classes)

    def test_first_stop_call(self):
        stopping_criterion = KappaAverage(2)

        predictions = np.array([0, 1, 1, 0])

        stop = stopping_criterion.stop(predictions=predictions)
        self.assertFalse(stop)

        self.assertEqual([], stopping_criterion.kappa_history)
        assert_array_equal(predictions, stopping_criterion.last_predictions)

    def test_second_stop_call(self):
        stopping_criterion = KappaAverage(2)

        first_predictions = np.array([0, 1, 1, 0])
        second_predictions = np.array([0, 1, 1, 1])

        stopping_criterion.stop(predictions=first_predictions)
        stop = stopping_criterion.stop(predictions=second_predictions)
        self.assertFalse(stop)

        self.assertEqual([0.5], stopping_criterion.kappa_history)
        assert_array_equal(second_predictions, stopping_criterion.last_predictions)

    def test_stop(self):
        stopping_criterion = KappaAverage(2, kappa=0.7)

        first_predictions = np.array([0, 1, 1, 0])
        second_predictions = np.array([0, 1, 1, 1])
        third_predictions = np.array([0, 1, 1, 1])
        fourth_predictions = np.array([0, 1, 1, 1])

        self.assertFalse(stopping_criterion.stop(predictions=first_predictions))
        self.assertFalse(stopping_criterion.stop(predictions=second_predictions))
        self.assertFalse(stopping_criterion.stop(predictions=third_predictions))
        self.assertTrue(stopping_criterion.stop(predictions=fourth_predictions))

        self.assertEqual([0.5, 1.0, 1.0], stopping_criterion.kappa_history)
        assert_array_equal(fourth_predictions, stopping_criterion.last_predictions)

    def test_stop_with_nan_values(self):
        stopping_criterion = KappaAverage(2, kappa=0.7)

        first_predictions = np.array([1, 1, 1, 1])
        second_predictions = np.array([1, 1, 1, 1])
        third_predictions = np.array([1, 1, 1, 1])
        fourth_predictions = np.array([1, 1, 1, 1])

        self.assertFalse(stopping_criterion.stop(predictions=first_predictions))
        self.assertFalse(stopping_criterion.stop(predictions=second_predictions))
        self.assertFalse(stopping_criterion.stop(predictions=third_predictions))
        self.assertTrue(stopping_criterion.stop(predictions=fourth_predictions))

        self.assertTrue(np.all(np.isnan(stopping_criterion.kappa_history)))
        assert_array_equal(fourth_predictions, stopping_criterion.last_predictions)

    def test_stop_with_predictions_none(self):
        stopping_criterion = KappaAverage(2)

        with self.assertRaises(ValueError):
            stopping_criterion.stop()

        with self.assertRaises(ValueError):
            stopping_criterion.stop(predictions=np.array([None, None, None]))

    def test_stop_with_prediction_size_changing(self):
        stopping_criterion = KappaAverage(2)

        with self.assertRaises(ValueError):
            stopping_criterion.stop(predictions=np.array([0, 1, 0]))
            stopping_criterion.stop(predictions=np.array([0, 1, 0, 1]))
