import numpy as np

from abc import ABC
from small_text.training.metrics import Metric


class ModelSelectionResult(object):
    """Results from a model selection.

    The epoch number starts at 1, unless no data for model selection is available,
    then a result with `epoch=0` is intended to represent an "empty response".

    There may be several models per epoch, therefore a unique id for every model is required.
    """

    def __init__(self, model_id, model_path, measured_values, fields=dict()):
        """
        Parameters
        ----------
        model_id : str
            Unique identifier for this model.
        model_path : str
            Path to the model.
        measured_values : dict of str to object
            A dictionary of measured values.
        fields : dict of str to object
            A dictionary of additional measured fields.
        """
        self.model_id = model_id
        self.model_path = model_path

        self.measured_values = measured_values
        self.fields = fields

    def __repr__(self):
        return f'ModelSelectionResult(\'{self.model_id}\', \'{self.model_path}\', ' \
               f'{self.measured_values}, {self.fields})'


class ModelSelectionManager(ABC):

    def add_model(self, model_id, model_path, measured_values, step=0, fields=dict()):
        """Adds the data for a trained model. This includes measured values of certain metrics
        and additional fields by which a model selection strategy then selects the model.

        Parameters
        ----------
        model_id : str
            Unique identifier for this model.
        model_path : str
            Path to the model.
        measured_values : dict of str to object
            A dictionary of measured values.
        step : int
            The number of the epoch (1-indexed) which is associated with this model.
        fields : dict of str to object
            A dictionary of additional measured fields.
        """

    def select(self, select_by=None):
        """Selects the best model.

        Parameters
        ----------
        select_by : str or list of str
            Name of the strategy that chooses the model. The choices are specific to the
            implementation.

        Returns
        -------
        model_selection_result : ModelSelectionResult or None
            A model selection result object which contains the data of the selected model
            or None if no model could be selected.
        """
        pass


class NoopModelSelection(ModelSelectionManager):
    """A no-operation model selection handler which. This is for developer
    convenience only, you will likely not need this in an application setting.

    .. versionadded:: 1.1.0
    """
    def __init__(self):
        self.last_model_id = None

    def add_model(self, model_id, model_path, measured_values, step=0, fields=dict()):
        _unused = step, model_path, measured_values, fields  # noqa:F841
        self.last_model_id = model_id

    def select(self, select_by=None):
        _unused = select_by  # noqa:F841
        return None


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

    FIELD_NAME_EARLY_STOPPING = 'early_stop'
    """Field name for the early stopping default field."""

    DEFAULT_DEFAULT_SELECT_BY = ['val_loss', 'val_acc']  # default "default_select_value" setting
    """Metrics by which the `select()` function chooses the best model."""

    def __init__(self, default_select_by=DEFAULT_DEFAULT_SELECT_BY, metrics=DEFAULT_METRICS,
                 required=DEFAULT_REQUIRED_METRIC_NAMES, fields_config=dict()):
        """
        Parameters
        ----------
        default_select_by : str or list of str
            Metric or list of metrics. In case a list is given, the model selection starts with
            the first metric and switches to the next one in case of a tie.
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
        if ModelSelection.FIELD_NAME_EARLY_STOPPING in fields_config:
            raise ValueError(f'Name conflict for field {ModelSelection.FIELD_NAME_EARLY_STOPPING} '
                             f'which already exists as a default field.')

        if isinstance(default_select_by, str):
            default_select_by = [default_select_by]
        self._verify_select_by(metrics, required, default_select_by)
        self.default_select_by = default_select_by

        self.metrics = metrics
        self.required = set(required)
        self._verify_metrics(self.metrics, required)

        self._fields_config = {**fields_config, **{ModelSelection.FIELD_NAME_EARLY_STOPPING: bool}}

        names = ['model_id', 'model_path'] \
            + [metric.name for metric in self.metrics] \
            + list(self._fields_config.keys())
        formats = [object, object] \
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

    @staticmethod
    def _verify_select_by(metrics, required, select_by):
        configured_metrics = np.union1d([metric.name for metric in metrics], required)
        setdiff = np.setdiff1d(select_by, configured_metrics)
        if configured_metrics.shape[0] > 0 and setdiff.shape[0] > 0:
            raise ValueError(f'Invalid metric(s) in select_by: {setdiff.tolist()}')

    def add_model(self, model_id, model_path, measured_values, fields=dict()):
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

        row = np.array((model_id, model_path) + tuple_measured_values, dtype=self.dtype)
        self.models = np.append(self.models, row)
        self.last_model_id = model_id

        return model_id

    def select(self, select_by=None):
        """
        Parameters
        ----------
        select_by : str or list of str
            Metric or list of metrics. Takes precedence over `self.default_select_by` if not None.
            In case a list is given, the model selection starts with the first metric and
            switches to the next one in case of a tie.

        Returns
        -------
        model_selection_result : ModelSelectionResult or None
            A model selection result object which contains the data of the selected model
            or None if no model could be selected.
        """
        if select_by is not None:
            if isinstance(select_by, str):
                select_by = [select_by]
        else:
            select_by = self.default_select_by

        # valid rows are rows where no early stopping has been triggered
        valid_rows = np.not_equal(self.models[ModelSelection.FIELD_NAME_EARLY_STOPPING], True)
        if not np.any(valid_rows):  # checks if we have no valid rows
            return None

        rows = self.models[valid_rows]

        metrics_dict = {metric.name: metric for metric in self.metrics}
        tuples = tuple(
            rows[key] if metrics_dict[key].lower_is_better else -rows[key]
            for key in reversed(select_by)
        )
        indices = np.lexsort(tuples)

        model_id = rows['model_id'][indices[0]]
        model_path = rows['model_path'][indices[0]]

        measured_values = {metric.name: rows[metric.name][indices[0]] for metric in self.metrics}
        fields = {key: rows[key][indices[0]] for key in self._fields_config}

        return ModelSelectionResult(model_id, model_path, measured_values, fields=fields)
