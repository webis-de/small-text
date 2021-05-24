import unittest

import pytest

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from active_learning.integrations.pytorch.query_strategies import (
        ExpectedGradientLength, ExpectedGradientLengthMaxWord)
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
class ExpectedGradientLengthTest(unittest.TestCase):

    def test_init_default(self):
        strategy = ExpectedGradientLength(2)

        self.assertEqual(2, strategy.num_classes)
        self.assertEqual(50, strategy.batch_size)
        self.assertEqual('cuda', strategy.device)

    def test_init(self):
        strategy = ExpectedGradientLength(4, batch_size=100, device='cpu')

        self.assertEqual(4, strategy.num_classes)
        self.assertEqual(100, strategy.batch_size)
        self.assertEqual('cpu', strategy.device)


@pytest.mark.pytorch
class ExpectedGradientLengthMaxWordTest(unittest.TestCase):

    def test_init_default(self):
        strategy = ExpectedGradientLengthMaxWord(2, 'embedding')

        self.assertEqual(2, strategy.num_classes)
        self.assertEqual(50, strategy.batch_size)
        self.assertEqual('cuda', strategy.device)
        self.assertEqual('embedding', strategy.layer_name)

    def test_init(self):
        strategy = ExpectedGradientLengthMaxWord(4, 'embedding', batch_size=100, device='cpu')

        self.assertEqual(4, strategy.num_classes)
        self.assertEqual(100, strategy.batch_size)
        self.assertEqual('cpu', strategy.device)
        self.assertEqual('embedding', strategy.layer_name)
