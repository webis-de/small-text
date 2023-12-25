import numpy as np
import numpy.typing as npt

from typing import List, Union

from small_text.data.sampling import _get_class_histogram

from small_text.query_strategies.base import constraints, ClassificationType, QueryStrategy


# The naming here is kept general (distributions and categories), but this is currently used to create distributions
#  over the number of classes
def _sample_distribution(num_samples: int,
                         source_distribution: npt.NDArray[np.uint],
                         ignored_values: List[int] = []):
    """Return a balanced sample from the given `source_distribution` of size ``num-samples`. The sample is represented
    in the form of an empirical categorial frequency distribution (i.e. a histogram). It is built iteratively, and
    prefers the category currently having the smallest number of samples.

    Parameters
    ----------
    num_samples : int
        Number of samples that the resulting distribution has.
    source_distribution : np.ndarray[int]
        A source frequency distribution in the shape of (num_values,) where num_values is the number of possible values
        for the source_distribution.
    ignored_values : list of int
        List of values (indices in the interval [0, `source_distribution.shape[0]`]) that should be ignored.

    Returns
    -------
    output_distribution : np.ndarray[int]
        A new distribution, which is  whose categories are less than or equal to the source distribution.
    """

    num_classes = source_distribution.shape[0]
    active_classes = np.array([i for i in range(num_classes) if i not in set(ignored_values)])

    new_distribution = np.zeros((num_classes,), dtype=int)
    for _ in range(num_samples):
        distribution_difference = (new_distribution - source_distribution)[active_classes]
        minima = np.where(distribution_difference == distribution_difference.min())[0]

        # Sample the class which occurs the least. In the case of a tie, the decision is random.
        if minima.shape[0] == 1:
            new_distribution[active_classes[minima[0]]] += 1
        else:
            sampled_minimum_index = np.random.choice(minima, 1)[0]
            new_distribution[active_classes[sampled_minimum_index]] += 1

    return new_distribution


def _get_rebalancing_distribution(num_samples, num_classes, y, y_pred, ignored_classes=[]):
    current_class_distribution = _get_class_histogram(y, num_classes)
    predicted_class_distribution = _get_class_histogram(y_pred, num_classes)

    number_per_class_required_for_balanced_dist = current_class_distribution.max() - current_class_distribution

    number_per_class_required_for_balanced_dist[list(ignored_classes)] = 0

    # Balancing distribution: When added to current_class_distribution, the result is balanced.
    optimal_balancing_distribution = current_class_distribution.max() - current_class_distribution
    target_distribution = _sample_distribution(num_samples,
                                               optimal_balancing_distribution,
                                               ignored_values=ignored_classes)

    # balancing_distribution:
    balancing_distribution = np.zeros((num_classes,), dtype=int)
    active_classes = np.array([i for i in range(num_classes) if i not in set(ignored_classes)])

    for c in active_classes:
        if predicted_class_distribution[c] < target_distribution[c]:
            # adapt the balancing distribution so that it can be sampled
            balancing_distribution[c] = predicted_class_distribution[c]
        else:
            balancing_distribution[c] = target_distribution[c]

    # The predicted labels does not have enough classes so that a sample with the desired balancing distribution
    # cannot be provided. Try to fill the remainder with other samples from "active classes" instead.
    remainder = target_distribution.sum() - balancing_distribution.sum()
    if remainder > 0:
        current_class_distribution += balancing_distribution

        free_active_class_samples = []
        for c in active_classes:
            class_indices = np.argwhere(y_pred == c)[:, 0]
            if class_indices.shape[0] > current_class_distribution[c]:
                free_active_class_samples.extend([c] * (class_indices.shape[0] - current_class_distribution[c]))

        np.random.shuffle(free_active_class_samples)
        for c in free_active_class_samples[:remainder]:
            balancing_distribution[c] += 1
            current_class_distribution[c] += 1

    # When not enough samples can be taken from the active classes, we fall back to using all classes.
    remainder = target_distribution.sum() - balancing_distribution.sum()
    if remainder > 0:
        free_ignored_class_samples = []
        for i, count in enumerate(predicted_class_distribution - balancing_distribution):
            if count > 0:
                free_ignored_class_samples.extend([i] * count)

        np.random.shuffle(free_ignored_class_samples)
        for c in free_ignored_class_samples[:remainder]:
            balancing_distribution[c] += 1

    return balancing_distribution


@constraints(classification_type=ClassificationType.SINGLE_LABEL)
class ClassBalancer(QueryStrategy):
    """A query strategy that tries to draw instances so that the new class distribution of the labeled pool
    is moved towards a (more) balanced distribution. For this, it first partitions instances by their
    predicted class and then applies a base query strategy. Based on the per-class query results, the
    instances are sampled so that the new class distribution is more balanced.

    Since the true labels are unknown, this strategy is a best effort approach and is not guaranteed
    to improve the distribution's balance.

    To reduce the cost of the initial predictions, which are required for the class-based partitioning,
    a random subsampling parameter is available.

    .. note ::
       The sampling mechanism is tailored to single-label classification.

    .. versionadded:: 2.0.0
    """

    def __init__(self, base_query_strategy: QueryStrategy, ignored_classes: List[int] = [],
                 subsample_size: Union[int, None] = None):
        """
        base_query_strategy : QueryStrategy
            A base query strategy which operates on the subsets partitioned by predicted class.
        subsample_size : int or None
            Draws a random subsample before applying the strategy if not `None`.
        """
        self.base_query_strategy = base_query_strategy
        self.ignored_classes = ignored_classes
        self.subsample_size = subsample_size

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        if self.subsample_size is None or self.subsample_size > indices_unlabeled.shape[0]:
            indices = indices_unlabeled
        else:
            indices_all = np.arange(indices_unlabeled.shape[0])
            indices_subsample = np.random.choice(indices_all,
                                                 self.subsample_size,
                                                 replace=False)
            indices = indices_unlabeled[indices_subsample]

        return self._query_class_balanced(clf, dataset, indices, indices_labeled, y, n)

    def _query_class_balanced(self, clf, dataset, indices, indices_labeled, y, n):
        y_pred = clf.predict(dataset[indices])

        target_distribution = _get_rebalancing_distribution(n,
                                                            clf.num_classes,
                                                            y,
                                                            y_pred,
                                                            ignored_classes=self.ignored_classes)

        active_classes = np.array([i for i in range(clf.num_classes) if i not in set(self.ignored_classes)])

        indices_balanced = []
        for c in active_classes:
            class_indices = np.argwhere(y_pred == c)[:, 0]
            if target_distribution[c] > 0:
                class_reduced_indices = np.append(indices[class_indices], indices_labeled)
                queried_indices = self.base_query_strategy.query(clf,
                                                                 dataset[class_reduced_indices],
                                                                 np.arange(class_indices.shape[0]),
                                                                 np.arange(class_indices.shape[0],
                                                                           class_reduced_indices.shape[0]),
                                                                 y,
                                                                 n=target_distribution[c])
                indices_balanced.extend(class_reduced_indices[queried_indices].tolist())

        return np.array(indices_balanced)

    def __str__(self):
        return f'ClassBalancer(base_query_strategy={self.base_query_strategy}, ' \
               f'ignored_classes={self.ignored_classes}, ' \
               f'subsample_size={self.subsample_size})'
