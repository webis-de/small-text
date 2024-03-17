import unittest
import pytest

import numpy as np

from unittest import mock

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier, KimCNNEmbeddingMixin
    from tests.utils.datasets import trec_dataset
    from tests.utils.testing_pytorch import autocast_asserting_decorator
except PytorchNotFoundError:
    pass


def default_module_selector(m):
    return m['fc']


class _KimCNNEmbeddingTest(object):

    def test_embed_model_not_trained(self):
        _, train, tokenizer = trec_dataset()  # use small test set as train

        embedding_matrix = torch.Tensor(np.random.rand(len(tokenizer.get_vocab()), 100))
        classifier = KimCNNClassifier(6, embedding_matrix=embedding_matrix)

        def module_selector(m):
            return m['fc']

        with self.assertRaises(ValueError):
            classifier.embed(train, module_selector=module_selector, embedding_method=self.embedding_method)

    def test_embed(self):
        _, train, tokenizer = trec_dataset()  # use small test set as train

        kwargs = dict()
        if self.embedding_method == KimCNNEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
            kwargs['module_selector'] = default_module_selector

        embedding_matrix = torch.Tensor(np.random.rand(len(tokenizer.get_vocab()), 100))
        classifier = KimCNNClassifier(6, embedding_matrix=embedding_matrix)
        classifier.fit(train)

        create_embeddings_spy = autocast_asserting_decorator(classifier._create_embeddings, False, self)
        with mock.patch.object(classifier,
                               '_create_embeddings',
                               wraps=create_embeddings_spy):

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
        _, train, tokenizer = trec_dataset()  # use small test set as train

        kwargs = dict()

        if self.embedding_method == KimCNNEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
            kwargs['module_selector'] = default_module_selector

        embedding_matrix = torch.Tensor(np.random.rand(len(tokenizer.get_vocab()), 100))
        classifier = KimCNNClassifier(6, embedding_matrix=embedding_matrix)
        classifier.fit(train)

        create_embeddings_spy = autocast_asserting_decorator(classifier._create_embeddings, False, self)
        with mock.patch.object(classifier,
                               '_create_embeddings',
                               wraps=create_embeddings_spy):

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

    def test_embed_with_amp_enabled(self):
        _, train, tokenizer = trec_dataset()  # use small test set as train

        kwargs = dict()
        if self.embedding_method == KimCNNEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
            kwargs['module_selector'] = default_module_selector

        embedding_matrix = torch.Tensor(np.random.rand(len(tokenizer.get_vocab()), 100))
        amp_args = AMPArguments(use_amp=True, device_type='cuda', dtype=torch.bfloat16)
        classifier = KimCNNClassifier(6,
                                      embedding_matrix=embedding_matrix,
                                      amp_args=amp_args)
        classifier.fit(train)

        create_embeddings_spy = autocast_asserting_decorator(classifier._create_embeddings, True, self)
        with mock.patch.object(classifier,
                               '_create_embeddings',
                               wraps=create_embeddings_spy):

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


@pytest.mark.pytorch
class KimCNNEmbeddingPooledTest(unittest.TestCase, _KimCNNEmbeddingTest):

    def setUp(self):
        self.embedding_method = KimCNNEmbeddingMixin.EMBEDDING_METHOD_POOLED


@pytest.mark.pytorch
class KimCNNEmbeddingGradientTest(unittest.TestCase, _KimCNNEmbeddingTest):

    def setUp(self):
        self.embedding_method = KimCNNEmbeddingMixin.EMBEDDING_METHOD_GRADIENT
