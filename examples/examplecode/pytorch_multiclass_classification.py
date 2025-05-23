"""Example of a multiclass active learning text classification with pytorch.
"""

import numpy as np

from small_text import (
    EmptyPoolException,
    BreakingTies,
    KimCNNClassifierFactory,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    random_initialization_stratified
)

from examplecode.data.example_data_multiclass import (
    get_train_test,
    preprocess_data,
    load_pretrained_word_vectors
)
from examplecode.shared import evaluate


def main(num_iterations=10, device='cuda'):
    from small_text.integrations.pytorch.classifiers.base import AMPArguments

    words, pretrained_vectors = load_pretrained_word_vectors()

    # loads a 3-class subset of 20news
    train, test = get_train_test()
    train, test, tokenizer, pretrained_vectors = preprocess_data(train, test, words, pretrained_vectors)

    num_classes = len(np.unique(train.y))

    # Active learning parameters
    classifier_kwargs = {
        'embedding_matrix': pretrained_vectors,
        'max_seq_len': 512,
        'num_epochs': 4,
        'device': device,
        'amp_args': AMPArguments(use_amp=True, device_type='cuda')
    }

    clf_factory = KimCNNClassifierFactory(num_classes, classifier_kwargs)
    query_strategy = BreakingTies()

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
    # Perform 20 iterations of active learning...
    for i in range(num_iterations):
        # ...where each iteration consists of labelling 20 samples
        indices_queried = active_learner.query(num_samples=20, representation=train)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print('Iteration #{:d} ({} samples)'.format(i, len(indices_labeled)))
        evaluate(active_learner, train[indices_labeled], test)


def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_stratified(y_train, 20)
    active_learner.initialize(indices_initial)

    return indices_initial


if __name__ == '__main__':
    import argparse
    import logging
    logging.getLogger('small_text').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='An example that shows active learning '
                                                 'for multi-class text classification.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='number of active learning iterations')

    args = parser.parse_args()

    main(num_iterations=args.num_iterations)
