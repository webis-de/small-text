import unittest
import pytest

from unittest import mock
from unittest.mock import patch

from parameterized import parameterized_class

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import twenty_news_transformers
from tests.utils.testing import assert_array_not_equal

try:
    import torch

    from torch.optim import Adadelta
    from transformers import get_linear_schedule_with_warmup

    from small_text.integrations.transformers import TransformerModelArguments
    from small_text.integrations.transformers.classifiers import TransformerBasedClassificationFactory, TransformerBasedClassification
    from small_text.integrations.transformers.classifiers.classification import FineTuningArguments, _get_layer_params
    from small_text.integrations.transformers.datasets import TransformersDataset

    from tests.utils.datasets import random_transformer_dataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
@parameterized_class([{'embedding_method': 'avg', 'num_classes': 2},
                      {'embedding_method': 'cls', 'num_classes': 2},
                      {'embedding_method': 'avg', 'num_classes': 3},
                      {'embedding_method': 'cls', 'num_classes': 3}])
class EmbeddingTest(unittest.TestCase):

    def test_embed_model_not_fitted(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            'sshleifer/tiny-distilroberta-base',
            self.num_classes,
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()

        with self.assertRaises(ValueError):
            clf.embed(train_set)

    def test_embed(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95),
            'num_epochs': 1
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            self.num_classes,
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        with mock.patch.object(clf.model,
                               'eval',
                               wraps=clf.model.eval) as model_eval_spy:

            embeddings = clf.embed(train_set, embedding_method=self.embedding_method)
            model_eval_spy.assert_called()

        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(len(train_set), embeddings.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embeddings.shape[1])

    def test_embed_with_layer_index(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95),
            'num_epochs': 1
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            self.num_classes,
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        with mock.patch.object(clf.model,
                               'eval',
                               wraps=clf.model.eval) as model_eval_spy:
            embedding_one = clf.embed(train_set, embedding_method=self.embedding_method)
            model_eval_spy.assert_called()

        with mock.patch.object(clf.model,
                               'eval',
                               wraps=clf.model.eval) as model_eval_spy:
            embedding_two = clf.embed(train_set, embedding_method=self.embedding_method,
                                  hidden_layer_index=0)
            model_eval_spy.assert_called()

        assert_array_not_equal(embedding_one, embedding_two)

        self.assertEqual(2, len(embedding_one.shape))
        self.assertEqual(2, len(embedding_two.shape))
        self.assertEqual(len(train_set), embedding_one.shape[0])
        self.assertEqual(len(train_set), embedding_two.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embedding_one.shape[1])
        self.assertEqual(clf.model.config.hidden_size, embedding_two.shape[1])

    def test_embed_with_predictions(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95),
            'num_epochs': 1
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            self.num_classes,
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        embeddings, predictions = clf.embed(train_set, return_proba=True, embedding_method=self.embedding_method)
        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(len(train_set), embeddings.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embeddings.shape[1])
        self.assertEqual(len(train_set), predictions.shape[0])


@pytest.mark.pytorch
class ClassificationTest(unittest.TestCase):

    @patch.object(TransformerBasedClassification, '_train')
    @patch.object(TransformerBasedClassification, '_select_best_model')
    def test_fit_distilroberta(self, select_best_model_mock, fake_train):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            2,
            kwargs=classifier_kwargs)

        x = twenty_news_transformers(20, num_labels=2)

        clf = clf_factory.new()
        clf.fit(x)

        # basically tests _get_layer_params for now

        fake_train.assert_called()
        select_best_model_mock.assert_called()

    @patch.object(TransformerBasedClassification, '_train')
    @patch.object(TransformerBasedClassification, '_select_best_model')
    def test_test_fit_distilbert(self, select_best_model_mock, fake_train):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('distilbert-base-cased'),
            2,
            kwargs=classifier_kwargs)

        x = twenty_news_transformers(20, num_labels=2)

        clf = clf_factory.new()
        clf.fit(x)

        # basically tests _get_layer_params for now

        fake_train.assert_called()
        select_best_model_mock.assert_called()

    def test_initialize_optimizer_and_scheduler_default(self):
        sub_train = random_transformer_dataset(10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2,
                                                    class_weight='balanced')

        base_lr = 2e-5
        fine_tuning_arguments = FineTuningArguments(base_lr, 0.95)

        optimizer = None
        scheduler = 'linear'
        params = None

        # initializes the model
        classifier.initialize_transformer(None)

        optimizer, scheduler = classifier._initialize_optimizer_and_scheduler(optimizer,
                                                                              scheduler,
                                                                              fine_tuning_arguments,
                                                                              base_lr,
                                                                              params,
                                                                              classifier.model,
                                                                              sub_train)

        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(scheduler)

        optimizer_params = [param for param in classifier.model.parameters() if param.requires_grad]
        self.assertEqual(optimizer.__class__,
                         classifier._default_optimizer(optimizer_params, base_lr).__class__)

    def test_initialize_optimizer_and_scheduler_custom(self):
        sub_train = random_transformer_dataset(10)

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        classifier = TransformerBasedClassification(model_args, 2,
                                                    class_weight='balanced')

        base_lr = 2e-5
        fine_tuning_arguments = FineTuningArguments(base_lr, 0.95)

        # initializes the model
        classifier.initialize_transformer(None)

        optimizer_params = [param for param in classifier.model.parameters() if param.requires_grad]
        optimizer_arg = Adadelta(optimizer_params)
        scheduler_arg = get_linear_schedule_with_warmup(optimizer_arg,
                                                        num_warmup_steps=10,
                                                        num_training_steps=100)
        params = None

        optimizer, scheduler = classifier._initialize_optimizer_and_scheduler(optimizer_arg,
                                                                              scheduler_arg,
                                                                              fine_tuning_arguments,
                                                                              base_lr,
                                                                              params,
                                                                              classifier.model,
                                                                              sub_train)

        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(scheduler)

        self.assertEqual(optimizer, optimizer_arg)
        self.assertEqual(scheduler, scheduler_arg)
