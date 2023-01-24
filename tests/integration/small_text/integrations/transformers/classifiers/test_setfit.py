import unittest
import pytest
import numpy as np

from scipy.sparse import issparse

from small_text.exceptions import UnsupportedOperationException
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from sklearn.utils.validation import check_is_fitted
from tests.utils.datasets import twenty_news_text

try:
    from small_text.integrations.transformers.classifiers.factories import (
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

        embeddings, proba = clf.embed(train_set, return_proba=True)

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

    def test_fit_and_validate(self):
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

        valid_set = twenty_news_text(10, num_classes=self.num_classes, multi_label=self.multi_label)

        if self.use_differentiable_head:
            with self.assertRaises(NotImplementedError):
                clf.validate(valid_set)
        else:
            with self.assertRaises(UnsupportedOperationException):
                clf.validate(valid_set)


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
