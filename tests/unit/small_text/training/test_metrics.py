import unittest

from small_text.training.metrics import Metric


class MetricsTest(unittest.TestCase):

    def test_repr(self):
        metric = Metric('train_acc', dtype=float, lower_is_better=False)
        self.assertEqual('Metric(\'train_acc\', dtype=float, lower_is_better=False)', repr(metric))
