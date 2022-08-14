import unittest

from small_text.training.metrics import Metric
from small_text.training.model_selection import (
    ModelSelectionResult, ModelSelection, NoopModelSelection
)


class ModelSelectionResultTest(unittest.TestCase):

    def test_init(self):
        epoch = 1
        model_id = '1'
        model_path = 'any'
        measured_values = {'val_loss': 0.23, 'val_acc': 0.81}

        model_selection_result = ModelSelectionResult(epoch, model_id, model_path, measured_values)

        self.assertEqual(epoch, model_selection_result.epoch)
        self.assertEqual(model_id, model_selection_result.model_id)
        self.assertEqual(model_path, model_selection_result.model_path)
        self.assertEqual(measured_values, model_selection_result.measured_values)

    def test_init_with_fields(self):
        epoch = 1
        model_id = '1'
        model_path = 'any'
        measured_values = {'val_loss': 0.23, 'val_acc': 0.81}
        fields = {'early_stopping': False}

        model_selection_result = ModelSelectionResult(epoch, model_id, model_path, measured_values,
                                                      fields=fields)

        self.assertEqual(epoch, model_selection_result.epoch)
        self.assertEqual(model_id, model_selection_result.model_id)
        self.assertEqual(model_path, model_selection_result.model_path)
        self.assertEqual(measured_values, model_selection_result.measured_values)
        self.assertEqual(fields, model_selection_result.fields)


class NoopModelSelectionTest(unittest.TestCase):

    def test_init(self):
        model_selection = NoopModelSelection()
        self.assertIsNone(model_selection.last_model_id)

    def test_add_model(self):
        model_selection = NoopModelSelection()

        measured_values = {}
        model_id = str(1)
        model_selection.add_model(1, model_id, '/any/path/to/model.bin', measured_values)
        self.assertEqual(model_id, model_selection.last_model_id)

    def test_select(self):
        model_selection = NoopModelSelection()
        self.assertIsNone(model_selection.last_model_id)

        model_selection_result = model_selection.select()

        self.assertEqual(0, model_selection_result.epoch)
        self.assertEqual(0, len(model_selection_result.measured_values))
        self.assertEqual(0, len(model_selection_result.fields))

        model_id = str(0)
        model_selection.add_model(1, model_id, '/any/path/to/model.bin', {})

        model_selection_result = model_selection.select()
        self.assertEqual(0, model_selection_result.epoch)
        self.assertEqual(0, len(model_selection_result.measured_values))
        self.assertEqual(0, len(model_selection_result.fields))


class ModelSelectionTest(unittest.TestCase):

    def test_init(self):
        model_selection = ModelSelection()
        self.assertIsNotNone(model_selection.metrics)
        self.assertIsNone(model_selection.last_model_id)
        self.assertEqual(4, len(model_selection.metrics))

    def test_init_with_empty_list_of_metrics(self):
        # not sure if it is a good idea but for now this will not be prevented
        model_selection = ModelSelection(metrics=[], required=[])
        self.assertIsNone(model_selection.last_model_id)
        self.assertEqual(0, len(model_selection.metrics))

    def test_init_invalid_metric_configuration(self):
        metrics = [Metric('val_acc', lower_is_better=False)]
        with self.assertRaisesRegex(ValueError, 'Required metric "val_loss" is not within'):
            ModelSelection(metrics=metrics, required=['val_loss'])

    def test_init_with_fields_config(self):
        fields_config = {'a': int}
        model_selection = ModelSelection(fields_config=fields_config)
        self.assertEqual(2, len(model_selection._fields_config))

    def test_init_with_fields_config_invalid_field(self):
        fields_config = {ModelSelection.EARLY_STOPPING_FIELD: bool}
        with self.assertRaisesRegex(ValueError, 'Name conflict for field'):
            ModelSelection(fields_config=fields_config)

    def test_add_model(self):
        model_selection = ModelSelection()

        measured_values = {'val_loss': 0.043, 'val_acc': 0.78,
                           'train_loss': 0.023, 'train_acc': 0.85}
        model_id = str(1)
        model_selection.add_model(1, model_id, '/any/path/to/model.bin', measured_values)
        self.assertEqual((1,), model_selection.models.shape)
        self.assertEqual(model_id, model_selection.last_model_id)

    def test_add_model_invalid_epoch(self):
        model_selection = ModelSelection()

        measured_values = {'val_loss': 0.043, 'val_acc': 0.78,
                           'train_loss': 0.023, 'train_acc': 0.85}
        with self.assertRaisesRegex(ValueError, 'Argument "epoch" must be greater than zero'):
            model_selection.add_model(0, str(1), '/any/path/to/model.bin', measured_values)

    def test_add_model_missing_metrics(self):
        model_selection = ModelSelection()

        # val_loss is missing
        measured_values = {'val_acc': 0.78, 'train_loss': 0.023, 'train_acc': 0.85}
        with self.assertRaisesRegex(ValueError,
                                    'Required measured values missing for metric "val_loss"'):
            model_selection.add_model(1, str(1), '/any/path/to/model.bin', measured_values)

    def test_add_model_with_fields_config(self):
        model_selection = ModelSelection(fields_config={'early_stop': bool})

        measured_values = {'val_loss': 0.043, 'val_acc': 0.78,
                           'train_loss': 0.023, 'train_acc': 0.85}
        fields = {'early_stop': True}
        model_selection.add_model(1, str(1), '/any/path/to/model.bin', measured_values,
                                  fields=fields)
        self.assertEqual((1,), model_selection.models.shape)
        self.assertTrue(model_selection.models[0]['early_stop'])

    def test_add_model_with_duplicate_id(self):
        model_selection = ModelSelection(fields_config={'early_stop': bool})

        measured_values = {'val_loss': 0.043, 'val_acc': 0.78,
                           'train_loss': 0.023, 'train_acc': 0.85}

        model_id = str(1)
        model_selection.add_model(1, model_id, '/any/path/to/model_1.bin', measured_values)
        with self.assertRaisesRegex(ValueError, 'Duplicate model_id'):
            model_selection.add_model(2, model_id, '/any/path/to/model_2.bin', measured_values)

    def test_add_model_with_duplicate_model_path(self):
        model_selection = ModelSelection(fields_config={'early_stop': bool})

        measured_values = {'val_loss': 0.043, 'val_acc': 0.78,
                           'train_loss': 0.023, 'train_acc': 0.85}

        model_path = '/any/path/to/model_1.bin'
        model_selection.add_model(1, str(1), model_path, measured_values)
        with self.assertRaisesRegex(ValueError, 'Duplicate model_path'):
            model_selection.add_model(2, str(2), model_path, measured_values)

    def test_select(self):
        model_selection = ModelSelection()

        measured_values_list = [
            {'val_loss': 0.043, 'val_acc': 0.78, 'train_loss': 0.023, 'train_acc': 0.85},
            {'val_loss': 0.033, 'val_acc': 0.79, 'train_loss': 0.020, 'train_acc': 0.85},
            {'val_loss': 0.031, 'val_acc': 0.77, 'train_loss': 0.018, 'train_acc': 0.85},
            {'val_loss': 0.032, 'val_acc': 0.78, 'train_loss': 0.016, 'train_acc': 0.85}
        ]
        for i, measured_values in enumerate(measured_values_list):
            model_selection.add_model(i+1, str(i+1), f'/any/path/to/model_{i+1}.bin',
                                      measured_values)

        model_selection_result = model_selection.select()

        self.assertEqual(3, model_selection_result.epoch)
        self.assertEqual('3', model_selection_result.model_id)
        self.assertEqual('/any/path/to/model_3.bin', model_selection_result.model_path)

        self.assertEqual(4, len(model_selection_result.measured_values))
        for key, val in measured_values_list[2].items():
            self.assertEqual(val, model_selection_result.measured_values[key])

        self.assertEqual(1, len(model_selection_result.fields))
        self.assertFalse(model_selection_result.fields['early_stopping'])

    def test_select_with_early_stopping(self):
        model_selection = ModelSelection()

        measured_values_list = [
            {'val_loss': 0.043, 'val_acc': 0.78, 'train_loss': 0.023, 'train_acc': 0.85},
            {'val_loss': 0.031, 'val_acc': 0.77, 'train_loss': 0.018, 'train_acc': 0.85},
            {'val_loss': 0.032, 'val_acc': 0.78, 'train_loss': 0.016, 'train_acc': 0.85},
            {'val_loss': 0.031, 'val_acc': 0.79, 'train_loss': 0.020, 'train_acc': 0.85}
        ]
        fields = [
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: True}
        ]
        for i, measured_values in enumerate(measured_values_list):
            model_selection.add_model(i+1, str(i+1), f'/any/path/to/model_{i+1}.bin',
                                      measured_values, fields=fields[i])

        model_selection_result = model_selection.select()

        self.assertEqual(4, len(model_selection_result.measured_values))
        for key, val in measured_values_list[1].items():
            self.assertEqual(val, model_selection_result.measured_values[key])

        self.assertEqual(1, len(model_selection_result.fields))
        self.assertFalse(model_selection_result.fields['early_stopping'])

    def test_select_with_early_stopping_select_by_second_metric(self):
        model_selection = ModelSelection()

        measured_values_list = [
            {'val_loss': 0.043, 'val_acc': 0.78, 'train_loss': 0.023, 'train_acc': 0.85},
            {'val_loss': 0.031, 'val_acc': 0.77, 'train_loss': 0.018, 'train_acc': 0.85},
            {'val_loss': 0.031, 'val_acc': 0.78, 'train_loss': 0.018, 'train_acc': 0.85},
            {'val_loss': 0.033, 'val_acc': 0.79, 'train_loss': 0.020, 'train_acc': 0.85},
        ]
        fields = [
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: True}
        ]
        for i, measured_values in enumerate(measured_values_list):
            model_selection.add_model(i+1, str(i+1), f'/any/path/to/model_{i+1}.bin',
                                      measured_values, fields=fields[i])

        model_selection_result = model_selection.select()

        self.assertEqual(4, len(model_selection_result.measured_values))
        for key, val in measured_values_list[2].items():
            self.assertEqual(val, model_selection_result.measured_values[key])

        self.assertEqual(1, len(model_selection_result.fields))
        self.assertFalse(model_selection_result.fields['early_stopping'])

    def test_select_with_early_stopping_select_tie_decide_by_epoch(self):
        model_selection = ModelSelection()

        measured_values_list = [
            {'val_loss': 0.043, 'val_acc': 0.78, 'train_loss': 0.023, 'train_acc': 0.85},
            {'val_loss': 0.031, 'val_acc': 0.77, 'train_loss': 0.018, 'train_acc': 0.85},
            {'val_loss': 0.031, 'val_acc': 0.77, 'train_loss': 0.018, 'train_acc': 0.85},
            {'val_loss': 0.033, 'val_acc': 0.79, 'train_loss': 0.020, 'train_acc': 0.85},
        ]
        fields = [
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: True}
        ]
        for i, measured_values in enumerate(measured_values_list):
            model_selection.add_model(i+1, str(i+1), f'/any/path/to/model_{i+1}.bin',
                                      measured_values, fields=fields[i])

        model_selection_result = model_selection.select()

        self.assertEqual(4, len(model_selection_result.measured_values))
        for key, val in measured_values_list[1].items():
            self.assertEqual(val, model_selection_result.measured_values[key])

        self.assertEqual(1, len(model_selection_result.fields))
        self.assertFalse(model_selection_result.fields['early_stopping'])

    def test_select_with_early_stopping_select_with_select_by_kwarg(self):
        model_selection = ModelSelection()

        measured_values_list = [
            {'val_loss': 0.043, 'val_acc': 0.78, 'train_loss': 0.023, 'train_acc': 0.86},
            {'val_loss': 0.031, 'val_acc': 0.77, 'train_loss': 0.018, 'train_acc': 0.85},
            {'val_loss': 0.032, 'val_acc': 0.77, 'train_loss': 0.018, 'train_acc': 0.86},
            {'val_loss': 0.033, 'val_acc': 0.79, 'train_loss': 0.020, 'train_acc': 0.86},
        ]
        fields = [
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: False},
            {ModelSelection.EARLY_STOPPING_FIELD: True}
        ]
        for i, measured_values in enumerate(measured_values_list):
            model_selection.add_model(i+1, str(i+1), f'/any/path/to/model_{i+1}.bin',
                                      measured_values, fields=fields[i])

        model_selection_result = model_selection.select(select_by=['train_acc', 'train_loss'])

        self.assertEqual(4, len(model_selection_result.measured_values))
        for key, val in measured_values_list[2].items():
            self.assertEqual(val, model_selection_result.measured_values[key])

        self.assertEqual(1, len(model_selection_result.fields))
        self.assertFalse(model_selection_result.fields['early_stopping'])
