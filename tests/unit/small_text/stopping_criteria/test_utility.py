import unittest

from small_text.stopping_criteria.utility import MaxIterations


class MaxIterationsTest(unittest.TestCase):

    def test_init(self):
        stopping_criterion = MaxIterations(2)

        self.assertEqual(2, stopping_criterion.max_iterations)
        self.assertEqual(0, stopping_criterion.current_iteration)

    def test_init_invalid_value(self):
        with self.assertRaisesRegex(ValueError, 'Argument max_iterations must be'):
            MaxIterations(0)

    def test_immedita_stop(self):
        stopping_criterion = MaxIterations(1)
        stop = stopping_criterion.stop()
        self.assertTrue(stop)

    def test_stop(self):
        stopping_criterion = MaxIterations(3)
        self.assertFalse(stopping_criterion.stop())
        self.assertFalse(stopping_criterion.stop())
        self.assertTrue(stopping_criterion.stop())
