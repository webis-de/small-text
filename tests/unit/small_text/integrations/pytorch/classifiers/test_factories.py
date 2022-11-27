import unittest
import pytest

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
    from small_text.integrations.pytorch.classifiers.factories import (
        KimCNNFactory
    )
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
class KimCNNFactoryTest(unittest.TestCase):

    def test_factory_new(self):
        vocab_length = 10
        embedding_matrix = torch.Tensor(np.random.rand(vocab_length, 100))
        factory = KimCNNFactory('kimcnn', 6, {'embedding_matrix': embedding_matrix})

        clf = factory.new()
        self.assertTrue(isinstance(clf, KimCNNClassifier))
        self.assertEqual(6, clf.num_classes)
