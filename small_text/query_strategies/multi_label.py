import numpy as np
import numpy.typing as npt

from typing import Union

from scipy.sparse import csr_matrix

from small_text.classifiers import Classifier
from small_text.data.datasets import Dataset
from small_text.query_strategies.base import ScoringMixin, constraints
from small_text.query_strategies.strategies import breaking_ties, QueryStrategy
from small_text.utils.context import build_pbar_context


def _validate_bounds(parameter_name: str, parameter_value: float):
    if parameter_value < 0.0 or parameter_value > 1.0:
        raise ValueError(f'{parameter_name} must be in the interval [0, 1].')
    return parameter_value


def _label_cardinality_inconsistency(y_pred_proba_unlabeled: npt.NDArray[np.double],
                                     y_labeled: csr_matrix,
                                     prediction_threshold: float = 0.5) -> float:
    """Computes the label cardinality inconsistency per instance [LG13]_.

    The label cardinality inconsistency is defined by the ($L^2$ norm of the) difference between the number of labels
    of an unlabeled instance and the expected number of labels according to the labeled set.

    Parameters
    ----------
    y_pred_unlabeled : np.ndarray[float]
        Confidence score distribution over all classes of shape (num_samples, num_classes).
    y_labeled : csr_matrix
       Labels of the instances in the labeled pool.
    prediction_threshold : float, default=0.5
       Once the prediction confidence ("proba") exceeds this threshold, the label counts as predicted.

    Returns
    -------
    label_cardinality_inconsistency : np.ndarray[float]
        A numpy array with the label cardinality inconsistency score for every unlabeled instance (i.e. with size
        `y_pred_unlabeled.shape[0]`).


    .. versionadded:: 2.0.0
    """
    if y_labeled.shape[0] == 0:
        raise ValueError('Labeled pool labels must not be empty.')

    average_count_per_labeled_instance = (y_labeled > 0).sum(axis=1).mean()
    count_per_unlabeled_instance = (y_pred_proba_unlabeled > prediction_threshold).sum(axis=1)
    count_per_unlabeled_instance = np.asarray(count_per_unlabeled_instance).ravel()

    label_cardinality_inconsistency = count_per_unlabeled_instance - average_count_per_labeled_instance

    return np.sqrt(np.power(label_cardinality_inconsistency, 2))


@constraints(classification_type='multi-label')
class LabelCardinalityInconsistency(ScoringMixin, QueryStrategy):
    """Queries the instances which exhibit the maximum label cardinality inconsistency [LG13]_.

    .. seealso::

        Function :py:func:`label_cardinality_inconsistency`.
            Function to compute the label cardinality inconsistency.

    .. versionadded:: 2.0.0
    """

    def __init__(self, prediction_threshold: float = 0.5):
        """
        Parameters
        ----------
        prediction_threshold : float, default=0.5
            Prediction threshold at which a confidence estimate is counted as a label.
        """
        self.prediction_threshold = _validate_bounds('Prediction threshold', prediction_threshold)
        self.scores_: Union[npt.NDArray[np.double], None] = None

    @property
    def last_scores(self):
        return self.scores_

    def score(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix]) -> npt.NDArray[np.double]:
        _unused = indices_labeled

        y_pred_proba_unlabeled = clf.predict_proba(dataset[indices_unlabeled])
        scores = _label_cardinality_inconsistency(y_pred_proba_unlabeled,
                                                  y,
                                                  prediction_threshold=self.prediction_threshold)
        self.scores_ = scores

        return scores

    def query(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix],
              n: int = 10) -> np.ndarray:
        self._validate_query_input(indices_unlabeled, n)

        scores = self.score(clf, dataset, indices_unlabeled, indices_labeled, y)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_queried = np.argpartition(-scores, n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_queried])

    def __str__(self):
        return f'LabelCardinalityInconsistency(prediction_threshold={self.prediction_threshold})'


# TODO: find a better name for uncertainty_weight before the release
def _uncertainty_weighted_label_cardinality_inconsistency(y_pred_proba_unlabeled: npt.NDArray[float],
                                                          y_labeled: csr_matrix,
                                                          uncertainty_weight: float = 0.5,
                                                          eps: float = 0.01) -> float:
    """Computes uncertainty-weighted label cardinality inconsistency per instance [LG13]_.

    The label cardinality inconsistency is defined by the ($L^2$ norm of the) difference between the number of labels
    of an unlabeled instance and the expected number of labels according to the labeled set.

    Parameters
    ----------
    y_pred_proba_unlabeled : npt.NDArray[float]
        Confidence score distribution over all classes of shape (num_samples, num_classes).
    y_labeled : csr_matrix
        Labels of the instances in the labeled pool.
    uncertainty_weight : float, default=0.5
        A weight between 0 and 1 that upweights the margin uncertainty score per label and downweights the
        label cardinality inconsistency per sample. Corresponds to the parameter $\beta$ in [LG13]_.
    eps : float, default=0.01
        A small value to be added to the denominator of the inverse uncertainty.

    Returns
    -------
    label_cardinality_inconsistency : np.ndarray[float]
        A numpy array with the label cardinality inconsistency score for every unlabeled instance (i.e. with size
        `y_pred_unlabeled.shape[0]`).


    .. versionadded:: 2.0.0
    """

    # +1 smoothing, so that in the case of a lci value of 0 the uncertainty is not nulled
    lci = _label_cardinality_inconsistency(y_pred_proba_unlabeled, y_labeled) + 1
    inverse_uncertainty = 1 / np.maximum(np.apply_along_axis(breaking_ties, 1, y_pred_proba_unlabeled), eps)

    return inverse_uncertainty ** uncertainty_weight * lci ** (1 - uncertainty_weight)


@constraints(classification_type='multi-label')
class AdaptiveActiveLearning(ScoringMixin, QueryStrategy):
    """Queries the instances which exhibit the maximum inverse margin uncertainty-weighted
    label cardinality inconsistency [LG13]_.

    This strategy is a combination of breaking ties and label cardinaly inconsistency.
    The keyword argument `uncertainty_weight` controls the weighting between those two.

    .. seealso::

        Function :py:func:`uncertainty_weighted_label_cardinality_inconsistency`.
            Function to compute uncertainty-weighted label cardinality inconsistency.

    .. versionadded:: 2.0.0
    """

    def __init__(self, uncertainty_weight: float = 0.5, prediction_threshold: float = 0.5):
        """
        Parameters
        ----------
        uncertainty_weight : float, default=0.5
            Weighting of the query strategy's uncertainty portion (between 0 and 1). A higher number
            prioritizes uncertainty, a lower number prioritizes label cardinality inconsistency.
        prediction_threshold : float, default=0.5
            Prediction threshold at which a confidence estimate is counted as a label.
        """
        self.prediction_threshold = _validate_bounds('Prediction threshold', prediction_threshold)
        self.uncertainty_weight = _validate_bounds('Uncertainty weight', uncertainty_weight)
        self.prediction_threshold = prediction_threshold

        self.scores_: Union[npt.NDArray[np.double], None] = None

    @property
    def last_scores(self):
        return self.scores_

    def score(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix]) -> npt.NDArray[np.double]:
        _unused = indices_labeled

        y_pred_proba_unlabeled = clf.predict_proba(dataset[indices_unlabeled])
        scores = _uncertainty_weighted_label_cardinality_inconsistency(y_pred_proba_unlabeled,
                                                                       y,
                                                                       uncertainty_weight=self.uncertainty_weight)

        self.scores_ = scores

        return scores

    def query(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix],
              n: int = 10) -> np.ndarray:
        self._validate_query_input(indices_unlabeled, n)

        scores = self.score(clf, dataset, indices_unlabeled, indices_labeled, y)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_queried = np.argpartition(-scores, n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_queried])

    def __str__(self):
        return f'AdaptiveActiveLearning(uncertainty_weight={self.uncertainty_weight}, ' \
               f'prediction_threshold={self.prediction_threshold})'


@constraints(classification_type='multi-label')
class CategoryVectorInconsistencyAndRanking(ScoringMixin, QueryStrategy):
    """Uncertainty Sampling based on Category Vector Inconsistency and Ranking of Scores [RCV18]_
    selects instances based on the inconsistency of predicted labels and per-class label rankings.
    """

    def __init__(self, batch_size: int = 2048, prediction_threshold: float = 0.5, epsilon: float = 1e-8, pbar='tqdm'):
        """
        Parameters
        ----------
        batch_size : int
            Batch size in which the computations are performed. Increasing the size increases
            the amount of memory used.
        prediction_threshold : float
            Confidence value above which a prediction counts as positive.
        epsilon : float
            A small value that is added to the argument of the logarithm to avoid taking the
            logarithm of zero.
        pbar : 'tqdm' or None, default='tqdm'
            Displays a progress bar if 'tqdm' is passed.
        """
        self.batch_size = batch_size
        self.prediction_threshold = prediction_threshold
        self.epsilon = epsilon
        self.pbar = pbar

        self.scores_: Union[npt.NDArray[np.double], None] = None

    @property
    def last_scores(self):
        return self.scores_

    def score(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix]) -> npt.NDArray[np.double]:
        _unused = indices_labeled

        y_proba = clf.predict_proba(dataset[indices_unlabeled])
        scores = self._compute_scores(indices_unlabeled, y, y_proba)

        self.scores_ = scores

        return scores

    def query(self,
              clf: Classifier,
              dataset: Dataset,
              indices_unlabeled: npt.NDArray[np.uint],
              indices_labeled: npt.NDArray[np.uint],
              y: Union[npt.NDArray[np.uint], csr_matrix],
              n: int = 10) -> np.ndarray:
        self._validate_query_input(indices_unlabeled, n)

        scores = self.score(clf, dataset, indices_unlabeled, indices_labeled, y)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_queried = np.argpartition(-scores, n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_queried])

    def _compute_scores(self, indices_unlabeled, y, proba):
        y_pred = (proba > self.prediction_threshold).astype(int)
        vector_inconsistency_scores = self._compute_vector_inconsistency(y,
                                                                         y_pred,
                                                                         proba.shape[1])
        ranking_scores = self._compute_ranking(indices_unlabeled, proba)
        return vector_inconsistency_scores * ranking_scores

    def _compute_vector_inconsistency(self, y, y_pred_unlabeled, num_classes):
        y_arr = y.toarray()

        num_batches = int(np.ceil(len(y_pred_unlabeled) / self.batch_size))

        vector_inconsistency = np.array([], dtype=np.float32)
        num_unlabeled = y_pred_unlabeled.shape[0]

        with build_pbar_context(self.pbar, tqdm_kwargs={'total': num_unlabeled}) as pbar:
            for batch_idx in np.array_split(np.arange(num_unlabeled), num_batches, axis=0):
                y_pred_unlabeled_sub = y_pred_unlabeled[batch_idx]
                # as an exception the variables a,b,c,d of the contingency table are adopted
                a = y_pred_unlabeled_sub.dot(y_arr.T)
                b = np.logical_not(y_pred_unlabeled_sub).dot(y_arr.T)
                c = y_pred_unlabeled_sub.dot(np.logical_not(y_arr).T)
                d = np.logical_not(y_pred_unlabeled_sub).dot(np.logical_not(y_arr).T).astype(int)

                hamming_distance = (b + c) / num_classes

                distance = self._distance(y_pred_unlabeled_sub, y_arr, num_classes,
                                          a, b, c, d, hamming_distance)
                distance = distance.sum(axis=1) / y_pred_unlabeled_sub.shape[0]
                vector_inconsistency = np.append(vector_inconsistency, distance)

                pbar.update(batch_idx.shape[0])

        return vector_inconsistency

    def _distance(self, y_pred_unlabeled_sub, y_arr, num_classes, a, b, c, d,
                  hamming_distance):

        distance = hamming_distance

        y_arr_ones = y_arr.sum(axis=1)
        y_arr_zeros = y_arr.shape[1] - y_arr_ones
        entropy_labeled = self._entropy(y_arr_ones, num_classes) \
            + self._entropy(y_arr_zeros, num_classes)
        entropy_labeled = np.tile(entropy_labeled[np.newaxis, :],
                                  (y_pred_unlabeled_sub.shape[0], 1))

        y_pred_unlabeled_sub_ones = y_pred_unlabeled_sub.sum(axis=1)
        y_pred_unlabeled_sub_zeros = y_pred_unlabeled_sub.shape[1] - y_pred_unlabeled_sub_ones
        entropy_unlabeled = self._entropy(y_pred_unlabeled_sub_ones, num_classes) \
            + self._entropy(y_pred_unlabeled_sub_zeros, num_classes)
        entropy_unlabeled = np.tile(entropy_unlabeled[:, np.newaxis], (1, y_arr.shape[0]))

        joint_entropy = self._entropy(b + c, num_classes) + self._entropy(a + d, num_classes)
        joint_entropy += (b + c) / num_classes \
            * (self._entropy(b, b + c)
               + self._entropy(c, b + c))
        joint_entropy += (a + d) / num_classes \
            * (self._entropy(a, a + d) + self._entropy(d, a + d))

        entropy_distance = 2 * joint_entropy - entropy_unlabeled - entropy_labeled
        entropy_distance /= (joint_entropy + self.epsilon)

        distance[hamming_distance == 1] = 1

        return distance

    def _entropy(self, numerator, denominator):
        ratio = numerator / (denominator + self.epsilon)
        result = -ratio * np.log2(ratio + self.epsilon)
        return result

    def _compute_ranking(self, indices_unlabeled, proba_unlabeled):
        num_unlabeled, num_classes = proba_unlabeled.shape[0], proba_unlabeled.shape[1]
        ranks = self._rank_by_margin(proba_unlabeled)

        ranking_denom = num_classes * (num_unlabeled - 1)

        ranking_scores = [
            sum([num_unlabeled - ranks[j, i]
                 for j in range(num_classes)]) / ranking_denom
            for i in range(indices_unlabeled.shape[0])
        ]
        return np.array(ranking_scores)

    def _rank_by_margin(self, proba):
        num_classes = proba.shape[1]

        proba_sum = proba.sum(axis=1)
        margin = proba - np.tile(proba_sum[:, np.newaxis], (1, num_classes))
        margin = np.absolute(margin)

        ranks = np.array([
            np.argsort(margin[:, j])
            for j in range(num_classes)
        ])
        return ranks

    def __str__(self):
        return f'CategoryVectorInconsistencyAndRanking(batch_size={self.batch_size}, ' \
               f'prediction_threshold={self.prediction_threshold}, ' \
               f'epsilon={self.epsilon})'
