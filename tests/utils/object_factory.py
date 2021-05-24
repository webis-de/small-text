import numpy as np

from active_learning.active_learner import PoolBasedActiveLearner

def get_initialized_active_learner(clf_factory, query_strategy, dataset):

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

    x_indices_initial = np.random.choice(np.arange(len(dataset)), size=10, replace=False)
    y_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    active_learner.initialize_data(x_indices_initial, y_initial)

    return active_learner
