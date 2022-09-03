import numpy as np

from small_text.active_learner import PoolBasedActiveLearner
from small_text.utils.labels import list_to_csr

from tests.utils.datasets import random_labeling


def get_initialized_active_learner(clf_factory, query_strategy, dataset, initial_indices=10,
                                   num_classes=2, multi_label=False):
    assert initial_indices > num_classes

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, dataset)

    indices_initial = np.random.choice(np.arange(len(dataset)), size=initial_indices, replace=False)

    if multi_label:
        y_initial = [[i] for i in np.arange(num_classes)] + \
                    [random_labeling(num_classes, multi_label=multi_label)
                     for _ in range(initial_indices - num_classes)]
        y_initial = list_to_csr(y_initial, (len(y_initial), num_classes))
    else:
        y_initial = np.append(np.arange(num_classes),
                              np.random.choice([0, 1], initial_indices-num_classes))
    active_learner.initialize_data(indices_initial, y_initial)

    return active_learner
