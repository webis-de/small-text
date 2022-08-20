import tempfile
import unittest

import pytest

from pathlib import Path

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError


try:
    from small_text.integrations.pytorch.models.kimcnn import KimCNN
    from small_text.integrations.pytorch.model_selection import Metric, PytorchModelSelection
except (ImportError, PytorchNotFoundError):
    pass


class MetricTest(unittest.TestCase):

    def test_metric_init(self):
        metric = Metric('valid_loss', lower_is_better=True)
        self.assertEqual('valid_loss', metric.name)
        self.assertTrue(metric.lower_is_better)

        metric = Metric('valid_acc', lower_is_better=False)
        self.assertEqual('valid_acc', metric.name)
        self.assertFalse(metric.lower_is_better)


@pytest.mark.pytorch
class ModelSelectionTest(unittest.TestCase):

    def test_model_selection_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            metrics = [Metric('valid_loss', lower_is_better=True),
                       Metric('valid_acc', lower_is_better=False),
                       Metric('train_loss', lower_is_better=True)]
            model_selection = PytorchModelSelection(Path(tmp_dir_name), metrics)

            self.assertEqual(model_selection.save_directory.absolute(),
                             Path(tmp_dir_name).absolute())

            self.assertIsNotNone(model_selection.metrics)
            self.assertEqual(3, len(model_selection.metrics))
            self.assertIsNotNone(model_selection.models)
            self.assertEqual(0, len(model_selection.models))

    def test_model_selection_add_model_missing_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            metrics = [Metric('valid_loss', lower_is_better=True),
                       Metric('valid_acc', lower_is_better=False),
                       Metric('train_loss', lower_is_better=True)]
            model_selection = PytorchModelSelection(Path(tmp_dir_name), metrics)

            with self.assertRaises(ValueError):
                model_selection.add_model(KimCNN(10, 20), 1)
