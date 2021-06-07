import unittest
import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNN
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class KimCNNIntegrationTest(unittest.TestCase):

    def test_simple_prediction(self):
        """
        Simple prediction with default weights (untrained).
        """

        n = 10
        vocab_size = 11
        num_classes = 3

        x = torch.randint(1, 11, (n, 98))
        x = torch.cat([x, torch.zeros((n, 2), dtype=torch.long)], dim=1)

        model = KimCNN(vocab_size, 100, num_classes=num_classes)

        output = model(x)

        self.assertEqual(n, output.size(0))
        self.assertEqual(num_classes, output.size(1))
