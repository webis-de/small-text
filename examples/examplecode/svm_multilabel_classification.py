"""Example of an svm multi-label active learning text classification.
"""
import numpy as np

from small_text.active_learner import PoolBasedActiveLearner
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory
from small_text.data.sampling import multilabel_stratified_subsets_sampling
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling

from examplecode.data.example_data_multilabel import (
    get_train_test,
    preprocess_data_sklearn as preprocess_data
)
from examplecode.shared import evaluate_multi_label


def main():
    # Prepare some data: The data is the go-emotions dataset (27 emotions + 1 neutral class)
    train, test = get_train_test()
    train, test = preprocess_data(train, test)
    num_classes = 28

    # Active learning parameters
    clf_template = ConfidenceEnhancedLinearSVC()
    clf_factory = SklearnClassifierFactory(clf_template,
                                           num_classes,
                                           kwargs=dict({'multi_label': True}))
    query_strategy = RandomSampling()

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
    """
    This is the main loop in which we perform 10 iterations of active learning.
    During each iteration 20 samples are queried and then updated.

    The update step reveals the true label to the active learner, i.e. this is a simulation,
    but in a real scenario the user input would be passed to the update function.
    """
    # Perform 10 iterations of active learning...
    for i in range(10):
        # ...where each iteration consists of labelling 20 samples
        queried_indices = active_learner.query(num_samples=100)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[queried_indices]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        labeled_indices = np.concatenate([queried_indices, labeled_indices])

        print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
        evaluate_multi_label(active_learner, train[labeled_indices], test)


def initialize_active_learner(active_learner, y_train):

    # Initialize the model - This is required for model-based query strategies.
    indices_initial = multilabel_stratified_subsets_sampling(y_train, n_samples=200)

    y_initial = y_train[indices_initial]

    active_learner.initialize_data(indices_initial, y_initial)

    return indices_initial


if __name__ == '__main__':
    main()
