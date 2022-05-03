import unittest
import pytest

import numpy as np

from unittest import mock

from parameterized import parameterized_class
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNEmbeddingMixin
    from tests.utils.datasets import trec_dataset
except PytorchNotFoundError:
    pass


def default_module_selector(m):
    return m['fc']


@pytest.mark.pytorch
@parameterized_class([{'embedding_method': 'pooled'},
                      {'embedding_method': 'gradient'}])
class KimCNNEmbeddingTest(unittest.TestCase):

    def test_embed_model_not_trained(self):

        _, train = trec_dataset()  # use small test set as train

        embedding_matrix = torch.Tensor(np.random.rand(len(train.vocab), 100))
        classifier = KimCNNClassifier(6, embedding_matrix=embedding_matrix)

        def module_selector(m):
            return m['fc']

        with self.assertRaises(ValueError):
            classifier.embed(train,
                             module_selector=module_selector,
                             embedding_method=self.embedding_method)

    def test_embed(self):

        _, train = trec_dataset()  # use small test set as train

        kwargs = dict()
        if self.embedding_method == KimCNNEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
            kwargs['module_selector'] = default_module_selector

        embedding_matrix = torch.Tensor(np.random.rand(len(train.vocab), 100))
        classifier = KimCNNClassifier(6, embedding_matrix=embedding_matrix)
        classifier.fit(train)

        with mock.patch.object(classifier.model,
                               'eval',
                               wraps=classifier.model.eval) as model_eval_spy:

            embeddings = classifier.embed(train,
                                          embedding_method=self.embedding_method,
                                          **kwargs)
            self.assertFalse(classifier.model.training)
            model_eval_spy.assert_called()

            self.assertEqual(len(train), embeddings.shape[0])
            if self.embedding_method == KimCNNEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
                gradient_length = classifier.model.out_channels * classifier.model.num_kernels \
                                  * classifier.model.num_classes
                self.assertEqual(classifier.num_classes * gradient_length,
                                 embeddings.shape[1])
            else:
                self.assertEqual(classifier.model.out_channels * classifier.model.num_kernels,
                                 embeddings.shape[1])

    def test_embed_and_predict(self):

        _, train = trec_dataset()  # use small test set as train

        kwargs = dict()

        if self.embedding_method == KimCNNEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
            kwargs['module_selector'] = default_module_selector

        embedding_matrix = torch.Tensor(np.random.rand(len(train.vocab), 100))
        classifier = KimCNNClassifier(6, embedding_matrix=embedding_matrix)
        classifier.fit(train)

        with mock.patch.object(classifier.model,
                               'eval',
                               wraps=classifier.model.eval) as model_eval_spy:

            embeddings, predictions = classifier.embed(train,
                                                       return_proba=True,
                                                       embedding_method=self.embedding_method,
                                                       **kwargs)

            self.assertFalse(classifier.model.training)
            model_eval_spy.assert_called()

            self.assertEqual(len(train), embeddings.shape[0])
            if self.embedding_method == KimCNNEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
                gradient_length = classifier.model.out_channels * classifier.model.num_kernels \
                                  * classifier.model.num_classes
                self.assertEqual(classifier.num_classes * gradient_length,
                                 embeddings.shape[1])
            else:
                self.assertEqual(classifier.model.out_channels * classifier.model.num_kernels,
                                 embeddings.shape[1])
            self.assertEqual(len(train), predictions.shape[0])
