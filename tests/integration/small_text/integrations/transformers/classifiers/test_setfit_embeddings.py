import unittest
import pytest

from unittest.mock import patch

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import twenty_news_text

try:
    import torch

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.transformers.classifiers.factories import (
        SetFitClassification,
        SetFitClassificationFactory
    )
    from small_text.integrations.transformers.classifiers.setfit import (
        SetFitModelArguments
    )
except (ImportError, PytorchNotFoundError):
    pass


class _EmbeddingTest(object):

    def test_embed_untrained(self):
        classification_kwargs = {'use_differentiable_head': self.use_differentiable_head}
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(20, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()

        with self.assertRaisesRegex(ValueError, 'Model is not trained'):
            clf.embed(train_set)

    def test_embed(self):
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2',
                                                 output_dir='/tmp')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(20, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()
        clf.fit(train_set)

        embeddings = clf.embed(train_set)

        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(len(train_set), embeddings.shape[0])
        # pooling layer is named '1'
        self.assertEqual(dict(clf.model.model_body.named_modules())['1'].pooling_output_dimension,
                         embeddings.shape[1])

    def test_embed_with_proba(self):
        device = 'cuda:0'
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label,
            'device': device
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2',
                                                 output_dir='/tmp')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(20, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()
        clf.fit(train_set)

        with patch.object(clf.model.model_body, 'encode', wraps=clf.model.model_body.encode) as encode_spy:
            embeddings, proba = clf.embed(train_set, return_proba=True)

            self.assertEqual(1, encode_spy.call_count)
            self.assertEqual(1, len(encode_spy.call_args_list[0].args))
            self.assertEqual(len(train_set.x), len(encode_spy.call_args_list[0].args[0]))

            if self.use_differentiable_head:
                self.assertEqual(2, len(encode_spy.call_args_list[0].kwargs))
                self.assertTrue(encode_spy.call_args_list[0].kwargs['convert_to_tensor'])
                self.assertEqual(device, encode_spy.call_args_list[0].kwargs['device'])
            else:
                self.assertEqual(1, len(encode_spy.call_args_list[0].kwargs))
                self.assertEqual(device, encode_spy.call_args_list[0].kwargs['device'])

            self.assertEqual(2, len(embeddings.shape))
            self.assertEqual(len(train_set), embeddings.shape[0])
            # pooling layer is named '1'
            self.assertEqual(dict(clf.model.model_body.named_modules())['1'].pooling_output_dimension,
                             embeddings.shape[1])

    def test_embed_with_amp_args(self):
        device = 'cuda:0'
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label,
            'device': device,
            'amp_args': AMPArguments(use_amp=True, device_type='cuda', dtype=torch.bfloat16),
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2',
                                                 output_dir='/tmp')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(20, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()
        clf.fit(train_set)

        with patch.object(clf.model.model_body, 'encode', wraps=clf.model.model_body.encode) as encode_spy:
            embeddings, proba = clf.embed(train_set, return_proba=True)

            self.assertEqual(1, encode_spy.call_count)
            self.assertEqual(1, len(encode_spy.call_args_list[0].args))
            self.assertEqual(len(train_set.x), len(encode_spy.call_args_list[0].args[0]))

            if self.use_differentiable_head:
                self.assertEqual(2, len(encode_spy.call_args_list[0].kwargs))
                self.assertTrue(encode_spy.call_args_list[0].kwargs['convert_to_tensor'])
                self.assertEqual(device, encode_spy.call_args_list[0].kwargs['device'])
            else:
                self.assertEqual(1, len(encode_spy.call_args_list[0].kwargs))
                self.assertEqual(device, encode_spy.call_args_list[0].kwargs['device'])

            self.assertEqual(2, len(embeddings.shape))
            self.assertEqual(len(train_set), embeddings.shape[0])
            # pooling layer is named '1'
            self.assertEqual(dict(clf.model.model_body.named_modules())['1'].pooling_output_dimension,
                             embeddings.shape[1])


@pytest.mark.pytorch
@pytest.mark.optional
class EmbeddingRegressionHeadSingleLabelTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = False
        self.use_differentiable_head = False


@pytest.mark.pytorch
@pytest.mark.optional
class EmbeddingRegressionHeadMultiLabelTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = True
        self.use_differentiable_head = False


@pytest.mark.pytorch
@pytest.mark.optional
class EmbeddingDifferentiableHeadSingleLabelTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = False
        self.use_differentiable_head = True


@pytest.mark.pytorch
@pytest.mark.optional
class EmbeddingDifferentiableHeadMultiLabelTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = True
        self.use_differentiable_head = True
