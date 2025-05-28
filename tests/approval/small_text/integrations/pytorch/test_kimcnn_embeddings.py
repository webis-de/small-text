import unittest
import pytest

import numpy as np

from approvaltests import verify

from examplecode.data.example_data_multiclass import (
    get_train_test,
    preprocess_data
)
from examplecode.pytorch_multiclass_classification import load_pretrained_word_vectors, initialize_active_learner

from small_text import KimCNNClassifierFactory, BreakingTies, PoolBasedActiveLearner
from small_text.integrations.pytorch.classifiers.base import AMPArguments
from tests.utils.misc import random_seed


@pytest.mark.pytorch
class PytorchKimCNNEmbeddingsApprovalTest(unittest.TestCase):

    @random_seed(seed=42, set_torch_seed=True)
    def test_embeddings(self, device='cuda'):
        train, test = get_train_test()
        words, pretrained_vectors = load_pretrained_word_vectors()

        train, test, tokenizer, pretrained_vectors = preprocess_data(train, test, words, pretrained_vectors)

        train = train[:100].clone()
        test = test[:10].clone()

        num_classes = len(np.unique(train.y))

        classifier_kwargs = {
            'embedding_matrix': pretrained_vectors,
            'max_seq_len': 256,
            'num_epochs': 2,
            'device': device
        }

        clf_factory = KimCNNClassifierFactory(num_classes, classifier_kwargs)
        query_strategy = BreakingTies()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
        initialize_active_learner(active_learner, train.y)

        with np.printoptions(precision=4, edgeitems=10, linewidth=np.inf):
            output = f'{active_learner.classifier.embed(test)}'
        verify(output)

    @random_seed(seed=42, set_torch_seed=True)
    def test_embeddings_amp(self, device='cuda'):
        train, test = get_train_test()
        words, pretrained_vectors = load_pretrained_word_vectors()

        train, test, tokenizer, pretrained_vectors = preprocess_data(train, test, words, pretrained_vectors)

        train = train[:100].clone()
        test = test[:10].clone()

        num_classes = len(np.unique(train.y))

        amp_args = AMPArguments(use_amp=True, device_type='cuda')
        classifier_kwargs = {
            'embedding_matrix': pretrained_vectors,
            'max_seq_len': 256,
            'num_epochs': 2,
            'device': device,
            'amp_args': amp_args
        }

        clf_factory = KimCNNClassifierFactory(num_classes, classifier_kwargs)
        query_strategy = BreakingTies()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
        initialize_active_learner(active_learner, train.y)

        with np.printoptions(precision=4, edgeitems=10, linewidth=np.inf):
            output = f'{active_learner.classifier.embed(test)}'
        verify(output)
