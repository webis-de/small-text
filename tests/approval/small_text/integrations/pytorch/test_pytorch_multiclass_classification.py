import unittest
import pytest

import numpy as np

from approvaltests import verify
from sklearn.metrics import f1_score

from examplecode.data.example_data_multiclass import (
    get_train_test,
    preprocess_data
)
from examplecode.pytorch_multiclass_classification import load_pretrained_word_vectors, initialize_active_learner

from small_text import KimCNNClassifierFactory, BreakingTies, PoolBasedActiveLearner
from tests.utils.misc import random_seed


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    output = f'Train: f1={f1_score(y_pred, train.y, average="micro"):.4f}\n'
    output += f'Test: f1={f1_score(y_pred_test, test.y, average="micro"):.4f}\n'

    return output


def perform_active_learning(active_learner, train, indices_labeled, test, num_iterations):

    output = ''

    for i in range(num_iterations):
        output += f'\nIteration {i+1}\n'
        indices_queried = active_learner.query(num_samples=10)

        y = train.y[indices_queried]
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        output += evaluate(active_learner, train[indices_labeled], test)

    return output


@pytest.mark.pytorch
class PytorchMulticlassClassificationApprovalTest(unittest.TestCase):

    @random_seed(seed=42, set_torch_seed=True)
    def test_kimcnn(self, device='cuda'):
        num_iterations = 3

        train, test = get_train_test()
        words, pretrained_vectors = load_pretrained_word_vectors()

        train, test, tokenizer, pretrained_vectors = preprocess_data(train, test, words, pretrained_vectors)

        num_classes = len(np.unique(train.y))

        classifier_kwargs = {
            'embedding_matrix': pretrained_vectors,
            'max_seq_len': 512,
            'num_epochs': 4,
            'device': device
        }

        clf_factory = KimCNNClassifierFactory(num_classes, classifier_kwargs)
        query_strategy = BreakingTies()

        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
        indices_labeled = initialize_active_learner(active_learner, train.y)

        with np.printoptions(edgeitems=10, linewidth=np.inf):
            output = f'{perform_active_learning(active_learner, train, indices_labeled, test, num_iterations)}'

        verify(output)
