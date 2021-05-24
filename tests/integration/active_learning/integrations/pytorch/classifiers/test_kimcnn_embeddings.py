import unittest
import pytest

import numpy as np

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from active_learning.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from tests.utils.datasets import trec_dataset
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
class KimCNNEmbeddingTest(unittest.TestCase):

    def test_embed_model_not_trained(self):

        _, train = trec_dataset()  # use small test set as train

        embedding_matrix = torch.Tensor(np.random.rand(len(train.vocab), 100))
        classifier = KimCNNClassifier(embedding_matrix=embedding_matrix)

        module_selector = lambda m: m['fc']
        with self.assertRaises(ValueError):
            classifier.embed(train, module_selector=module_selector)

    def test_embed(self):

        _, train = trec_dataset()  # use small test set as train

        embedding_matrix = torch.Tensor(np.random.rand(len(train.vocab), 100))
        classifier = KimCNNClassifier(embedding_matrix=embedding_matrix)

        classifier.fit(train)

        module_selector = lambda m: m['fc']
        embeddings = classifier.embed(train, module_selector=module_selector)

        self.assertEqual(len(train), embeddings.shape[0])
        gradient_length = classifier.model.out_channels * classifier.model.n_kernels \
                          * classifier.model.num_classes
        self.assertEqual(classifier.num_class * gradient_length,
                         embeddings.shape[1])
