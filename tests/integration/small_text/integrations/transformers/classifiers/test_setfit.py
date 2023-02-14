import unittest
import pytest
import numpy as np

from unittest.mock import patch
from scipy.sparse import issparse

from small_text.data.datasets import TextDataset
from small_text.exceptions import UnsupportedOperationException
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from sklearn.utils.validation import check_is_fitted
from tests.utils.datasets import twenty_news_text

try:
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
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
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
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
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

    def test_embed_untrained(self):
        with self.assertRaises(NotImplementedError):
            super().test_embed_untrained()

    def test_embed(self):
        with self.assertRaises(NotImplementedError):
            super().test_embed()

    def test_embed_with_proba(self):
        with self.assertRaises(NotImplementedError):
            super().test_embed_with_proba()


@pytest.mark.pytorch
@pytest.mark.optional
class EmbeddingDifferentiableHeadMultiLabelTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = True
        self.use_differentiable_head = True

    def test_embed_untrained(self):
        with self.assertRaises(NotImplementedError):
            super().test_embed_untrained()

    def test_embed(self):
        with self.assertRaises(NotImplementedError):
            super().test_embed()

    def test_embed_with_proba(self):
        with self.assertRaises(NotImplementedError):
            super().test_embed_with_proba()


class _ClassificationTest(object):

    def test_fit_with_misplaced_max_length_kwargs(self):
        setfit_model_args = SetFitModelArguments('sentence-transformers/all-MiniLM-L6-v2')
        num_classes = 5

        setfit_train_kwargs = {'max_length': 20}

        texts = ['this is a sentence', 'another sentence']
        clf = SetFitClassification(setfit_model_args, num_classes)
        dataset = TextDataset.from_arrays(texts, np.array([1, 0]), target_labels=np.array([0, 1]))
        with self.assertRaisesRegex(ValueError, 'Invalid keyword argument in setfit_train_kwargs'):
            clf.fit(dataset, setfit_train_kwargs=setfit_train_kwargs)

    def test_fit_and_predict(self):
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()
        clf.fit(train_set)
        if self.use_differentiable_head:
            self.assertTrue(check_is_fitted(clf.model.model_head))

        test_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)
        y_pred = clf.predict(test_set)

        if self.multi_label:
            self.assertEqual((30, self.num_classes), y_pred.shape)
            self.assertTrue(issparse(y_pred))
            self.assertEqual(y_pred.dtype, np.int64)
            self.assertTrue(np.logical_or(y_pred.indices.all() >= 0, y_pred.indices.all() <= 3))
        else:
            self.assertEqual((30,), y_pred.shape)
            self.assertTrue(isinstance(y_pred, np.ndarray))
            self.assertTrue(np.all([isinstance(y, np.int64) for y in y_pred]))
            self.assertTrue(np.logical_or(y_pred.all() >= 0, y_pred.all() <= 3))

        y_pred_proba = clf.predict_proba(test_set)
        self.assertEqual((30, self.num_classes), y_pred_proba.shape)
        self.assertTrue(isinstance(y_pred_proba, np.ndarray))
        self.assertTrue(np.all([isinstance(y, np.float64) for row in y_pred_proba for y in row]))
        self.assertTrue(np.logical_or(y_pred_proba.all() >= 0.0, y_pred_proba.all() <= 1.0))

    def test_fit_and_predict_proba_dropout(self, dropout_sampling=3):
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)
        test_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()

        if self.use_differentiable_head:
            with self.assertRaises(NotImplementedError):
                clf.fit(train_set)
        else:
            clf.fit(train_set)

            # check Module.train()/.eval() works
            clf.model.model_body.train()
            self.assertTrue(clf.model.model_body.training)
            clf.model.model_body.eval()
            self.assertFalse(clf.model.model_body.training)

            y_pred_proba = clf.predict_proba(test_set)
            self.assertSequenceEqual((len(test_set), self.num_classes), y_pred_proba.shape)
            self.assertTrue(isinstance(y_pred_proba, np.ndarray))
            self.assertTrue(np.all([isinstance(p, np.float64) for pred in y_pred_proba for p in pred]))

            y_pred_proba = clf.predict_proba(test_set, dropout_sampling=dropout_sampling)
            self.assertSequenceEqual((len(test_set), dropout_sampling, self.num_classes), y_pred_proba.shape)
            self.assertTrue(isinstance(y_pred_proba, np.ndarray))
            self.assertTrue(np.all([isinstance(p, np.float64) for pred in y_pred_proba
                                    for sample in pred for p in sample]))

            # check Module.train()/.eval() **still** (!) works
            clf.model.model_body.train()
            self.assertTrue(clf.model.model_body.training)
            clf.model.model_body.eval()
            self.assertFalse(clf.model.model_body.training)

    def test_fit_and_validate(self):
        device = 'cuda:0'
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label,
            'device': device
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()
        with patch.object(clf.model.model_body, 'to', wraps=clf.model.model_body.to) as to_spy:
            clf.fit(train_set)

            self.assertEqual(3, to_spy.call_count)
            # our call is the first
            self.assertEqual(1, len(to_spy.call_args_list[0].args))
            self.assertEqual(device, to_spy.call_args_list[0].args[0])

        valid_set = twenty_news_text(10, num_classes=self.num_classes, multi_label=self.multi_label)

        if self.use_differentiable_head:
            with self.assertRaises(NotImplementedError):
                clf.validate(valid_set)
        else:
            with self.assertRaises(UnsupportedOperationException):
                clf.validate(valid_set)

    def test_fit_with_non_default_settings(self):
        # in particularly we test max_seq_len and mini_batch_size here
        mini_batch_size = 8
        max_seq_len = 32
        device = 'cuda:0'
        classification_kwargs = {
            'use_differentiable_head': self.use_differentiable_head,
            'multi_label': self.multi_label,
            'device': device,
            'mini_batch_size': mini_batch_size,
            'max_seq_len': max_seq_len
        }
        setfit_model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf_factory = SetFitClassificationFactory(
            setfit_model_args,
            self.num_classes,
            classification_kwargs=classification_kwargs)

        train_set = twenty_news_text(30, num_classes=self.num_classes, multi_label=self.multi_label)

        clf = clf_factory.new()

        with patch('small_text.integrations.transformers.classifiers.setfit.SetFitTrainer',
                   autospec=True, create=True) as setfit_trainer_mock:
            clf.fit(train_set)

            self.assertEqual(1, setfit_trainer_mock.call_count)
            self.assertEqual(mini_batch_size, setfit_trainer_mock.call_args_list[0].kwargs['batch_size'])

            train_mock = setfit_trainer_mock.return_value.train
            self.assertEqual(1, train_mock.call_count)
            self.assertEqual(max_seq_len, train_mock.call_args_list[0].kwargs['max_length'])


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationRegressionHeadSingleLabelTest(unittest.TestCase, _ClassificationTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = False
        self.use_differentiable_head = False


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationDifferentiableHeadSingleLabelTest(unittest.TestCase, _ClassificationTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = False
        self.use_differentiable_head = True

    def test_fit_and_predict(self):
        with self.assertRaises(NotImplementedError):
            super().test_fit_and_predict()

    def test_fit_and_validate(self):
        with self.assertRaises(NotImplementedError):
            super().test_fit_and_validate()

    def test_fit_with_non_default_settings(self):
        with self.assertRaises(NotImplementedError):
            super().test_fit_with_non_default_settings()


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationRegressionHeadMultiLabelTest(unittest.TestCase, _ClassificationTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = True
        self.use_differentiable_head = False


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitClassificationDifferentiableHeadMultiLabelTest(unittest.TestCase, _ClassificationTest):

    def setUp(self):
        self.num_classes = 3
        self.multi_label = False
        self.use_differentiable_head = True

    def test_fit_and_predict(self):
        with self.assertRaises(NotImplementedError):
            super().test_fit_and_predict()

    def test_fit_and_validate(self):
        with self.assertRaises(NotImplementedError):
            super().test_fit_and_validate()

    def test_fit_with_non_default_settings(self):
        with self.assertRaises(NotImplementedError):
            super().test_fit_with_non_default_settings()
