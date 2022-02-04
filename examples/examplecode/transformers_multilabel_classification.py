"""Example of a transformer-based active learning multi-label text classification.
"""
import logging

import numpy as np

from transformers import AutoTokenizer

from small_text.active_learner import PoolBasedActiveLearner
from small_text.initialization import random_initialization_stratified
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling

from examplecode.data.example_data_multilabel import (
    get_train_test
)
from examplecode.data.example_data_transformers import preprocess_data
from examplecode.shared import evaluate_multi_label


TRANSFORMER_MODEL = TransformerModelArguments('distilroberta-base')


def main():
    # Active learning parameters
    num_classes = 28
    clf_factory = TransformerBasedClassificationFactory(TRANSFORMER_MODEL,
                                                        num_classes,
                                                        kwargs=dict({
                                                            'device': 'cuda',
                                                            'multi_label': True
                                                        }))
    query_strategy = RandomSampling()

    # Prepare some data
    train, test = get_train_test()

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL.model, cache_dir='.cache/')
    x_train = preprocess_data(tokenizer, train['text'], train['labels'], multi_label=True)

    x_test = preprocess_data(tokenizer, test['text'], test['labels'], multi_label=True)

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, x_train)
    labeled_indices = initialize_active_learner(active_learner, x_train.y)

    try:
        perform_active_learning(active_learner, x_train, labeled_indices, x_test)

    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def perform_active_learning(active_learner, train, labeled_indices, test):
    # Perform 10 iterations of active learning...
    for i in range(10):
        # ...where each iteration consists of labelling 20 samples
        q_indices = active_learner.query(num_samples=1000)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[q_indices]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        labeled_indices = np.concatenate([q_indices, labeled_indices])

        print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
        evaluate_multi_label(active_learner, train[labeled_indices], test)


def initialize_active_learner(active_learner, y_train):

    x_indices_initial = random_initialization_stratified(y_train, n_samples=2000)
    y_initial = y_train[x_indices_initial]

    active_learner.initialize_data(x_indices_initial, y_initial)

    return x_indices_initial


if __name__ == '__main__':
    logging.getLogger('small_text').setLevel(logging.INFO)
    logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

    main()
