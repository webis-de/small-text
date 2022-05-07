import numpy as np

from small_text.active_learner import PoolBasedActiveLearner


def get_initialized_active_learner(clf_factory, query_strategy, dataset, initial_indices=10,
                                   num_classes=2):
    assert initial_indices > num_classes

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

    indices_initial = np.random.choice(np.arange(len(dataset)), size=initial_indices, replace=False)
    y_initial = np.append(np.arange(num_classes),
                          np.random.choice([0, 1], initial_indices-num_classes))
    active_learner.initialize_data(indices_initial, y_initial)

    return active_learner
