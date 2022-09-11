import numpy as np

from small_text.query_strategies.base import constraints
from small_text.query_strategies.strategies import QueryStrategy
from small_text.utils.context import build_pbar_context


@constraints(classification_type='multi-label')
class CategoryVectorInconsistencyAndRanking(QueryStrategy):
    """Uncertainty Sampling based on Category Vector Inconsistency and Ranking of Scores [RCV18]_
    selects instances based on the inconsistency of predicted labels and per-class label rankings.
    """

    def __init__(self, batch_size=2048, prediction_threshold=0.5, epsilon=1e-8, pbar='tqdm'):
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

    def query(self, clf, dataset, indices_unlabeled, _indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        y_proba = clf.predict_proba(dataset[indices_unlabeled])
        scores = self._compute_scores(indices_unlabeled, y, y_proba)

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
