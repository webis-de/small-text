"""Example of a transformer-based active learning multi-class text classification.
"""
import logging

import numpy as np

from transformers import AutoTokenizer

from small_text.active_learner import PoolBasedActiveLearner
from small_text.initialization import random_initialization_balanced
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling

from examplecode.data.corpus_twenty_news import get_twenty_newsgroups_corpus
from examplecode.data.example_data_transformers import preprocess_data
from examplecode.shared import evaluate


TRANSFORMER_MODEL = TransformerModelArguments('distilroberta-base')

TWENTY_NEWS_SUBCATEGORIES = ['rec.sport.baseball', 'sci.med', 'rec.autos']


def main():
    # Active learning parameters
    num_classes = len(TWENTY_NEWS_SUBCATEGORIES)
    clf_factory = TransformerBasedClassificationFactory(TRANSFORMER_MODEL,
                                                        num_classes,
                                                        kwargs=dict({'device': 'cuda'}))
    query_strategy = RandomSampling()

    # Prepare some data
    train, test = get_twenty_newsgroups_corpus(categories=TWENTY_NEWS_SUBCATEGORIES)

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL.model, cache_dir='.cache/')
    train = preprocess_data(tokenizer, train.data, train.target)

    test = preprocess_data(tokenizer, test.data, test.target)

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    labeled_indices = initialize_active_learner(active_learner, train.y)

    try:
        perform_active_learning(active_learner, train, labeled_indices, test)

    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def perform_active_learning(active_learner, train, labeled_indices, test):
    # Perform 10 iterations of active learning...
    for i in range(10):
        # ...where each iteration consists of labelling 20 samples
        queried_indices = active_learner.query(num_samples=20)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[queried_indices]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        labeled_indices = np.concatenate([queried_indices, labeled_indices])

        print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
        evaluate(active_learner, train[labeled_indices], test)


def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_balanced(y_train)
    y_initial = np.array([y_train[i] for i in indices_initial])

    active_learner.initialize_data(indices_initial, y_initial)

    return indices_initial


if __name__ == '__main__':
    logging.getLogger('small_text').setLevel(logging.INFO)

    main()
