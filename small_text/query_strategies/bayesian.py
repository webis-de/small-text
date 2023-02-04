import numpy as np
from small_text.query_strategies.strategies import QueryStrategy


def _bald(p, eps=1e-8):
    p_mean = np.mean(p, axis=1)
    model_prediction_entropy = -np.sum(p_mean * np.log2(p_mean + eps), axis=-1)
    expected_prediction_entropy = -np.mean(np.sum(p * np.log2(p + eps), axis=-1), axis=1)
    return model_prediction_entropy - expected_prediction_entropy


class BALD(QueryStrategy):
    """Selects instances according to the Bayesian Active Learning by Disagreement (BALD) [HHG+11]_
    strategy.

    Requires that the `predict_proba()` method of the given classifier supports
    dropout sampling [GZ16]_.

    .. versionadded:: 1.2.0
    """
    def __init__(self, dropout_samples=10):
        """
        Parameters
        ----------
        dropout_samples : int
            For every instance in the dataset, `dropout_samples`-many predictions will be used
            to obtain uncertainty estimates.
        """
        self.dropout_samples = dropout_samples

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        proba_dopout_sampled = clf.predict_proba(dataset, dropout_sampling=self.dropout_samples)
        bald_scores = _bald(proba_dopout_sampled)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_partitioned = np.argpartition(-bald_scores[indices_unlabeled], n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_partitioned])

    def __str__(self):
        return f'BALD(dropout_samples={self.dropout_samples})'
