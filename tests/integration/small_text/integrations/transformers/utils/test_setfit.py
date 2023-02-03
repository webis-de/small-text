import unittest
import pytest
import numpy as np

from small_text.data.datasets import TextDataset
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.transformers.classifiers.factories import (
        SetFitClassification
    )
    from small_text.integrations.transformers.classifiers.setfit import (
        SetFitModelArguments
    )
    from small_text.integrations.transformers.utils.setfit import _truncate_texts
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
@pytest.mark.optional
class SetFitUtilsTest(unittest.TestCase):

    def test_truncate_texts(self):
        model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf = SetFitClassification(model_args, 3)

        dataset = TextDataset.from_arrays([
                'This is a very long text',
                'This is another very long text',
                'This is'
            ],
            np.array([0, 1, 2]),
            target_labels=np.arange(3)
        )
        datasets = [dataset]

        result_datasets = _truncate_texts(clf.model, 3, *datasets)
        self.assertEqual(1, len(result_datasets))
        self.assertEqual('this is a', result_datasets[0].x[0])
        self.assertEqual('this is another', result_datasets[0].x[1])
        self.assertEqual('this is', result_datasets[0].x[2])

    def test_truncate_texts_no_action_needed(self):
        model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf = SetFitClassification(model_args, 3)

        dataset = TextDataset.from_arrays([
                'This is a very long text',
                'This is another very long text',
                'This is'
            ],
            np.array([0, 1, 2]),
            target_labels=np.arange(3)
        )
        datasets = [dataset]

        result_datasets = _truncate_texts(clf.model, 20, *datasets)
        self.assertEqual(1, len(result_datasets))
        self.assertEqual(dataset.x[0], result_datasets[0].x[0])
        self.assertEqual(dataset.x[1], result_datasets[0].x[1])
        self.assertEqual(dataset.x[2], result_datasets[0].x[2])

    def test_truncate_texts_multiple_datasets(self):
        model_args = SetFitModelArguments('sentence-transformers/paraphrase-MiniLM-L3-v2')
        clf = SetFitClassification(model_args, 3)

        dataset = TextDataset.from_arrays([
                'This is a very long text',
                'This is another very long text',
                'This is'
            ],
            np.array([0, 2, 1]),
            target_labels=np.arange(3)
        )
        dataset_two = TextDataset.from_arrays([
                'Another text',
                'Yet another text'
            ],
            np.array([0, 2]),
            target_labels=np.arange(3)
        )
        datasets = [dataset, dataset_two]

        result_datasets = _truncate_texts(clf.model, 3, *datasets)
        self.assertEqual(2, len(result_datasets))
        self.assertEqual('this is a', result_datasets[0].x[0])
        self.assertEqual('this is another', result_datasets[0].x[1])
        self.assertEqual('this is', result_datasets[0].x[2])

        self.assertEqual('another text', result_datasets[1].x[0])
        self.assertEqual('yet another text', result_datasets[1].x[1])
