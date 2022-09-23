import unittest

import numpy as np
from small_text.stopping_criteria.uncertainty import OverallUncertainty


class OverallUncertaintyTest(unittest.TestCase):

    def test_init(self):
        stopping_criterion = OverallUncertainty(2)

        self.assertEqual(2, stopping_criterion.num_classes)
        self.assertEqual(0.05, stopping_criterion.threshold)

    def test_init_with_kwargs(self):

        num_classes = 2
        threshold = 0.1

        stopping_criterion = OverallUncertainty(num_classes, threshold=threshold)

        self.assertEqual(num_classes, stopping_criterion.num_classes)
        self.assertEqual(threshold, stopping_criterion.threshold)

    def test_init_with_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, 'Threshold must be'):
            OverallUncertainty(2, threshold=-1)
        with self.assertRaisesRegex(ValueError, 'Threshold must be'):
            OverallUncertainty(2, threshold=1)

    def test_first_stop_call(self):
        stopping_criterion = OverallUncertainty(2)

        proba = np.array([
            [0.5, 0.5]
        ])

        stop = stopping_criterion.stop(proba=proba, indices_stopping=np.arange(1))
        self.assertFalse(stop)

    def test_stop_single_prediction(self):
        stopping_criterion = OverallUncertainty(4)

        proba_first = np.array([
            [0, 1, 1, 0]
        ])
        proba_second = np.array([
            [0, 0, 1, 0]
        ])

        stop = stopping_criterion.stop(proba=proba_first, indices_stopping=np.arange(1))
        self.assertFalse(stop)
        stop = stopping_criterion.stop(proba=proba_second, indices_stopping=np.arange(1))
        self.assertTrue(stop)

    def test_stop_multiple_predictions(self):
        stopping_criterion = OverallUncertainty(4)

        proba_first = np.array([
            [0, 0.5, 0.5, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        proba_second = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0]
        ])

        stop = stopping_criterion.stop(proba=proba_first, indices_stopping=np.arange(3))
        self.assertFalse(stop)
        stop = stopping_criterion.stop(proba=proba_second, indices_stopping=np.arange(3))
        self.assertTrue(stop)

    def test_stop_with_proba_none(self):
        stopping_criterion = OverallUncertainty(2)

        with self.assertRaisesRegex(ValueError, 'indices_stopping must not be None'):
            stopping_criterion.stop()

        with self.assertRaisesRegex(ValueError, 'indices_stopping must not be None'):
            stopping_criterion.stop(proba=None)
