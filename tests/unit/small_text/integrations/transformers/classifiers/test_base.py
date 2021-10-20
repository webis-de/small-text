import pytest
import unittest

import numpy as np

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch

    from small_text.integrations.transformers.datasets import TransformersDataset
    from small_text.integrations.transformers.classifiers import TransformerBasedClassification
    from small_text.integrations.transformers.classifiers.classification import TransformerModelArguments
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


class _TransformerClassifierBaseFunctionalityTest(object):

    def _get_clf(self):
        raise NotImplementedError()

    def test_predict_on_empty_data(self):
        test_set = TransformersDataset([], None)

        clf = self._get_clf()
        # here would be a clf.fit call, which omit due to the runtime costs

        predictions = clf.predict(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_predict_proba_on_empty_data(self):
        test_set = TransformersDataset([], None)

        clf = self._get_clf()
        # here would be a clf.fit call, which omit due to the runtime costs

        predictions, proba = clf.predict_proba(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))
        self.assertEqual(0, proba.shape[0])
        self.assertTrue(np.issubdtype(proba.dtype, np.float))


@pytest.mark.pytorch
class TransformerBasedClassificationBaseFunctionalityTest(unittest.TestCase,_TransformerClassifierBaseFunctionalityTest):

    def _get_clf(self):
        return TransformerBasedClassification(TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
                                              2)
