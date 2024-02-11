"""Example of a transformer-based active learning multi-class text classification.
"""
import numpy as np

from transformers import AutoTokenizer

from small_text import (
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    RandomSampling,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    random_initialization_balanced
)

from examplecode.data.corpus_twenty_news import get_twenty_newsgroups_corpus
from examplecode.data.example_data_transformers import preprocess_data
from examplecode.shared import evaluate


TRANSFORMER_MODEL = TransformerModelArguments('distilroberta-base')

TWENTY_NEWS_SUBCATEGORIES = ['rec.sport.baseball', 'sci.med', 'rec.autos']


def main(num_iterations=10):
    # Active learning parameters
    num_classes = len(TWENTY_NEWS_SUBCATEGORIES)
    clf_factory = TransformerBasedClassificationFactory(TRANSFORMER_MODEL,
                                                        num_classes,
                                                        kwargs={
                                                            'device': 'cuda'
                                                        })
    query_strategy = RandomSampling()

    # Prepare some data
    train, test = get_twenty_newsgroups_corpus(categories=TWENTY_NEWS_SUBCATEGORIES)

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL.model, cache_dir='.cache/')
    train = preprocess_data(tokenizer, train.data, train.target)

    test = preprocess_data(tokenizer, test.data, test.target)

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    indices_labeled = initialize_active_learner(active_learner, train.y)

    try:
        perform_active_learning(active_learner, train, indices_labeled, test, num_iterations)
    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def perform_active_learning(active_learner, train, indices_labeled, test, num_iterations):
    # Perform 10 iterations of active learning...
    for i in range(num_iterations):
        # ...where each iteration consists of labelling 20 samples
        indices_queried = active_learner.query(num_samples=20)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print('Iteration #{:d} ({} samples)'.format(i, len(indices_labeled)))
        evaluate(active_learner, train[indices_labeled], test)


def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_balanced(y_train)
    active_learner.initialize(indices_initial)

    return indices_initial


if __name__ == '__main__':
    import argparse
    import logging
    logging.getLogger('small_text').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='An example that shows active learning '
                                                 'for multi-class text classification '
                                                 'using transformers.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='number of active learning iterations')

    args = parser.parse_args()

    main(num_iterations=args.num_iterations)
