"""Example of a binary active learning text classification.
"""
import numpy as np

from small_text import (
    ConfidenceEnhancedLinearSVC,
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    RandomSampling,
    SklearnClassifierFactory
)

from examplecode.data.example_data_binary import get_train_test, preprocess_data
from examplecode.shared import evaluate


def main(num_iterations=10):
    # Prepare some data: The data is a 2-class subset of 20news (baseball vs. hockey)
    text_train, text_test = get_train_test()
    train, test = preprocess_data(text_train, text_test)
    num_classes = 2

    # Active learning parameters
    clf_template = ConfidenceEnhancedLinearSVC()
    clf_factory = SklearnClassifierFactory(clf_template, num_classes)
    query_strategy = RandomSampling()

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    labeled_indices = initialize_active_learner(active_learner, train.y)

    try:
        perform_active_learning(active_learner, train, labeled_indices, test, num_iterations)
    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def perform_active_learning(active_learner, train, indices_labeled, test, num_iterations):
    """
    This is the main loop in which we perform 10 iterations of active learning.
    During each iteration 20 samples are queried and then updated.

    The update step reveals the true label to the active learner, i.e. this is a simulation,
    but in a real scenario the user input would be passed to the update function.
    """
    # Perform 10 iterations of active learning...
    for i in range(num_iterations):
        # ...where each iteration consists of labelling 20 samples
        indices_queried = active_learner.query(num_samples=20)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        print('Iteration #{:d} ({} samples)'.format(i, len(indices_labeled)))
        evaluate(active_learner, train[active_learner.indices_labeled], test)


def initialize_active_learner(active_learner, y_train):

    # Initialize the model. This is required for model-based query strategies.
    indices_pos_label = np.where(y_train == 1)[0]
    indices_neg_label = np.where(y_train == 0)[0]

    indices_initial = np.concatenate([np.random.choice(indices_pos_label, 10, replace=False),
                                      np.random.choice(indices_neg_label, 10, replace=False)],
                                     dtype=int)

    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='An example that shows active learning '
                                                 'for binary text classification.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='number of active learning iterations')

    args = parser.parse_args()

    main(num_iterations=args.num_iterations)
