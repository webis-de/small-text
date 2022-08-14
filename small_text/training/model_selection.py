import numpy as np

from abc import ABC
from small_text.training.metrics import Metric


class ModelSelectionResult(object):
    """Results from a model selection.

    The epoch number starts at 1, unless no data for model selection is available,
    then a result with `epoch=0` is intended to represent an "empty response".

    There may be several models per epoch, therefore a unique id for every model is required.
    """

    def __init__(self, epoch, model_id, model_path, measured_values, fields=dict()):
        """
        Parameters
        ----------
        epoch : int
            Epoch number which is associated with this model (1-indexed).
        model_id : str
            Unique identifier for this model.
        model_path : str
            Path to the model.
        measured_values : dict of str to object
            A dictionary of measured values.
        fields : dict of str to object
            A dictionary of additional measured fields.
        """
        self.epoch = epoch
        self.model_id = model_id
        self.model_path = model_path

        self.measured_values = measured_values
        self.fields = fields


class ModelSelectionManager(ABC):

    def add_model(self, epoch, model_id, model_path, measured_values, fields=dict()):
        """Adds the data for a trained model. This includes measured values of certain metrics
        and additional fields by which a model selection strategy then selects the model.

        Parameters
        ----------
        epoch : int
            The number of the epoch (1-indexed) which is associated with this model.
        model_id : str
            Unique identifier for this model.
        model_path : str
            Path to the model.
        measured_values : dict of str to object
            A dictionary of measured values.
        fields : dict of str to object
            A dictionary of additional measured fields.
        """

    def select(self, select_by=None):
        """Selects the best model.

        Returns
        -------
        model_selection_result : ModelSelectionResult
            A model selection result object which contains the data of the selected model.
        """
        pass


class NoopModelSelection(ModelSelectionManager):
    """A no-operation model selection handler which. This is for developer
    convenience only, you will likely not need this in an application setting.

    .. versionadded:: 1.1.0
    """
    def __init__(self):
        self.last_model_id = None

    def add_model(self, epoch, model_id, model_path, measured_values, fields=dict()):
        _unused = epoch, model_path, measured_values, fields  # noqa:F841
        self.last_model_id = model_id

    def select(self, select_by=None):
        _unused = select_by  # noqa:F841
        return ModelSelectionResult(0, self.last_model_id, None, {})


class ModelSelection(ModelSelectionManager):
    """A default model selection implementation.

    .. versionadded:: 1.1.0
    """

    DEFAULT_METRICS = [
        Metric('val_loss'),
        Metric('val_acc', lower_is_better=False),
        Metric('train_loss'),
        Metric('train_acc', lower_is_better=False)
    ]
    """Default metric configuration to be used."""

    DEFAULT_REQUIRED_METRIC_NAMES = ['val_loss', 'val_acc']
    """Names of the metrics that must be reported to add_model()."""

    EARLY_STOPPING_FIELD = 'early_stop'
    """Key for the early stopping default field."""

    DEFAULT_SELECT_BY = ['val_loss', 'val_acc']
    """Metrics by which the `select()` function chooses the best model."""

    def __init__(self, metrics=DEFAULT_METRICS, required=DEFAULT_REQUIRED_METRIC_NAMES,
                 fields_config=dict()):
        """
        Parameters
        ----------
        metrics : list of small_text.training.metrics.Metric
            The metrics whose measured values which will be used for deciding which model to use.
        required : list of str
            Names of the metrics given by `metrics` that are required.
            Non-required metrics can be reported as `None`.
        fields_config : dict of str to type
            A configuration for additional data fields that can be measured and taken
            into account when selecting the model. Fields can be None by default but can be
            required by model selection strategies.
        """
        if ModelSelection.EARLY_STOPPING_FIELD in fields_config:
            raise ValueError(f'Name conflict for field {ModelSelection.EARLY_STOPPING_FIELD} '
                             f'which already exists as a default field.')

        self.metrics = metrics
        self.required = set(required)
        self._verify_metrics(self.metrics, required)

        self._fields_config = {**fields_config, **{ModelSelection.EARLY_STOPPING_FIELD: bool}}

        names = ['epoch', 'model_id', 'model_path'] \
            + [metric.name for metric in self.metrics] \
            + list(self._fields_config.keys())
        formats = [int, object, object] \
            + [metric.dtype for metric in self.metrics] \
            + list(self._fields_config.values())

        self.dtype = {'names': names, 'formats': formats}

        self.last_model_id = None
        self.models = np.empty((0,), dtype=self.dtype)

    @staticmethod
    def _verify_metrics(metrics, required):
        metric_names = set([metric.name for metric in metrics])
        for required_metric_name in required:
            if required_metric_name not in metric_names:
                raise ValueError(f'Required metric "{required_metric_name}" is not within the '
                                 f'list of given metrics.')

    def add_model(self, epoch, model_id, model_path, measured_values, fields=dict()):
        if epoch <= 0:
            raise ValueError('Argument "epoch" must be greater than zero.')

        if (self.models['model_id'] == model_id).sum() > 0:
            raise ValueError(f'Duplicate model_id "{model_id}')
        elif (self.models['model_path'] == model_path).sum() > 0:
            raise ValueError(f'Duplicate model_path "{model_path}')

        for metric_name in self.required:
            if metric_name not in measured_values:
                raise ValueError(f'Required measured values missing for metric "{metric_name}"')

        tuple_measured_values = tuple(
            [measured_values.get(metric.name, None) for metric in self.metrics]
            + [fields.get(key, None) for key in self._fields_config]
        )

        row = np.array((epoch, model_id, model_path) + tuple_measured_values, dtype=self.dtype)
        self.models = np.append(self.models, row)
        self.last_model_id = model_id

        return model_id

    def select(self, select_by=DEFAULT_SELECT_BY):
        valid_rows = np.not_equal(self.models[ModelSelection.EARLY_STOPPING_FIELD], True)
        rows = self.models[valid_rows]

        metrics_dict = {metric.name: metric for metric in self.metrics}
        tuples = tuple([tuple(rows['epoch'])]) + \
            tuple(
                rows[key] if metrics_dict[key].lower_is_better else -rows[key]
                for key in reversed(select_by)
            )
        indices = np.lexsort(tuples)

        if indices.shape[0] == 0:
            return ModelSelectionResult(0, self.last_model_id, None, {})

        epoch = rows['epoch'][indices[0]]
        model_id = rows['model_id'][indices[0]]
        model_path = rows['model_path'][indices[0]]

        measured_values = {metric.name: rows[metric.name][indices[0]] for metric in self.metrics}
        fields = {key: rows[key][indices[0]] for key in self._fields_config}

        return ModelSelectionResult(epoch, model_id, model_path, measured_values, fields=fields)
