import json
import unittest
import warnings

from pathlib import Path

from numpy.testing import assert_array_equal

from active_learning.stopping_criteria.kappa import KappaAverage


class KappaAverageTest(unittest.TestCase):

    def test_initialization(self):
        stopping_criterion = KappaAverage()

        self.assertEqual([], stopping_criterion.kappa_history)
        self.assertEqual(3, stopping_criterion.window_size)
        self.assertEqual(0.99, stopping_criterion.kappa)

    def test_initialization_with_kwargs(self):

        window_size = 5
        kappa = 0.95

        stopping_criterion = KappaAverage(kappa=kappa, window_size=window_size)

        self.assertEqual([], stopping_criterion.kappa_history)
        self.assertEqual(window_size, stopping_criterion.window_size)
        self.assertEqual(kappa, stopping_criterion.kappa)

    def test_first_evaluation(self):
        stopping_criterion = KappaAverage()

        predictions = [0, 1, 1, 0]

        stop = stopping_criterion.evaluate(predictions)
        self.assertFalse(stop)

        self.assertEqual([], stopping_criterion.kappa_history)
        self.assertEqual(predictions, stopping_criterion.last_predictions)

    def test_second_evaluation(self):
        stopping_criterion = KappaAverage()

        first_predictions = [0, 1, 1, 0]
        second_predictions = [0, 1, 1, 1]

        stopping_criterion.evaluate(first_predictions)
        stop = stopping_criterion.evaluate(second_predictions)
        self.assertFalse(stop)

        self.assertEqual([0.5], stopping_criterion.kappa_history)
        self.assertEqual(second_predictions, stopping_criterion.last_predictions)

    def test_stop(self):
        stopping_criterion = KappaAverage(kappa=0.99)

        fixture_path = str(Path(__file__).parent.joinpath('fixture_predictions.json'))
        with open(fixture_path, 'r') as f:
            predictions = json.load(f)

        self.assertEqual(5, len(predictions))

        for i in range(4):
            self.assertFalse(stopping_criterion.evaluate(predictions[i]))

        stop = stopping_criterion.evaluate(predictions[4])
        self.assertTrue(stop)

        self.assertEqual([0.5, 0.5, 0.5], stopping_criterion.kappa_history)
        self.assertEqual(predictions[4], stopping_criterion.last_predictions)

    def test_stop_with_nans_occurring(self):
        stopping_criterion = KappaAverage(kappa=0.99)

        fixture_path = str(Path(__file__).parent.joinpath('fixture_predictions_nan.json'))
        with open(fixture_path, 'r') as f:
            predictions = json.load(f)

        self.assertEqual(5, len(predictions))

        for i in range(4):
            self.assertFalse(stopping_criterion.evaluate(predictions[i]))

        with warnings.catch_warnings(record=True) as w:
            stop = stopping_criterion.evaluate(predictions[4])
            self.assertTrue(stop)

            self.assertEqual(1, len(w))
            self.assertTrue(issubclass(w[0].category, RuntimeWarning))

        assert_array_equal([0.0, float('nan'), float('nan')], stopping_criterion.kappa_history)
        self.assertEqual(predictions[4], stopping_criterion.last_predictions)
