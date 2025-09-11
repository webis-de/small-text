import logging
import warnings

from abc import abstractmethod
from functools import partial
from pathlib import Path

from small_text.classifiers.classification import Classifier
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.training.early_stopping import (
    EarlyStopping,
    NoopEarlyStopping,
    EarlyStoppingOrCondition
)
from small_text.training.metrics import Metric
from small_text.training.model_selection import ModelSelection, NoopModelSelection
from small_text.utils.classification import empty_result, prediction_result

try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
    from torch.optim.lr_scheduler import LambdaLR

    from small_text.integrations.pytorch.utils.contextmanager import inference_mode
    from small_text.integrations.pytorch.utils.data import dataloader, get_class_weights
    from small_text.integrations.pytorch.utils.loss import _LossAdapter2DTo1D
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


logger = logging.getLogger(__name__)


def _check_optimizer_and_scheduler_config(optimizer, scheduler):
    if scheduler is not None and optimizer is None:
        raise ValueError('You must also pass an optimizer if you pass a scheduler to fit()')


class AMPArguments(object):
    """Arguments for configuring Automated Mixed Precision.

       .. seealso::

          `Pytorch Docs: Automatic Mixed Precision Package <https://pytorch.org/docs/stable/amp.html>`

          `PyTorch Docs: Automatic Mixed Precision Recipes <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>`

       .. versionadded:: 2.0.0
    """

    def __init__(self, use_amp: bool = False, device_type=None, dtype=torch.bfloat16):
        """
        use_amp : bool, default=False
            Enabled AMP if true.
        device_type : str
            Device type to be used for torch.autocast ('cuda' or 'cpu').
        dtype : torch.dtype, default=torch.bfloat16
            Data type to be used for torch.autocast (torch.float16 or torch.bfloat16).
        """
        self.use_amp = use_amp
        self.device_type = device_type
        self.dtype = dtype


class AMPMixin(object):

    @property
    def amp_args(self):
        if self._amp_args is None:
            device_type = 'cpu' if self.model is None else self.model.device.type
            amp_args = AMPArguments(device_type=device_type, dtype=torch.bfloat16)
        else:
            amp_args = AMPArguments(use_amp=self._amp_args.use_amp,
                                    device_type=self._amp_args.device_type,
                                    dtype=self._amp_args.dtype)
        if self.model is None or self.model.device.type == 'cpu':
            amp_args.use_amp = False
        return amp_args


class PytorchModelSelectionMixin(object):

    def _save_model(self, optimizer, model_selection, model_id, train_acc, train_loss,
                    valid_acc, valid_loss, stop, tmp_dir):

        measured_values = {'train_acc': train_acc, 'train_loss': train_loss,
                           'val_acc': valid_acc, 'val_loss': valid_loss}
        model_path = Path(tmp_dir).joinpath(f'model_{model_id}.pt')
        torch.save(self.model.state_dict(), model_path)
        optimizer_path = model_path.with_suffix('.pt.optimizer')
        torch.save(optimizer.state_dict(), optimizer_path)
        model_selection.add_model(model_id, model_path, measured_values,
                                  fields={ModelSelection.FIELD_NAME_EARLY_STOPPING: stop})

    def _perform_model_selection(self, optimizer, model_selection):
        model_selection_result = model_selection.select()
        if model_selection_result is not None:
            self.model.load_state_dict(torch.load(model_selection_result.model_path))
            optimizer_path = model_selection_result.model_path.with_suffix('.pt.optimizer')
            optimizer.load_state_dict(torch.load(optimizer_path))


class PytorchClassifier(PytorchModelSelectionMixin, AMPMixin, Classifier):

    def __init__(self, multi_label=False, device=None, mini_batch_size=32, amp_args=None):

        self.multi_label = multi_label
        self.mini_batch_size = mini_batch_size

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if self.device.startswith('cuda'):
            logging.info('torch.version.cuda: %s', torch.version.cuda)
            logging.info('torch.cuda.is_available(): %s', torch.cuda.is_available())
            if torch.cuda.is_available():
                logging.info('torch.cuda.current_device(): %s', torch.cuda.current_device())

        self.model = None
        self._amp_args = amp_args

    @abstractmethod
    def fit(self, train_set, validation_set=None, weights=None, **kwargs):
        pass

    def predict(self, data_set, return_proba=False, multi_label_threshold: float=0.5):
        """
        Parameters
        ----------
        data_set : small_text.data.Dataset
            A dataset on whose instances predictions are made.
        return_proba : bool
            If True, additionally returns the confidence distribution over all classes.

        Returns
        -------
        predictions : np.ndarray[np.int32] or csr_matrix[np.int32]
            List of predictions if the classifier was fitted on multi-label data,
            otherwise a sparse matrix of predictions.
        probas : np.ndarray[np.float32] (optional)
            List of probabilities (or confidence estimates) if `return_proba` is True.
        """
        if len(data_set) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=True,
                                return_proba=return_proba)

        proba = self.predict_proba(data_set)
        # TODO: For multi-label, we convert from csr -> ndarray -> csr. This can be made more efficient.
        predictions = prediction_result(proba.toarray() if self.multi_label else proba,
                                        self.multi_label,
                                        self.num_classes,
                                        multi_label_threshold=multi_label_threshold)

        if return_proba:
            return predictions, proba

        return predictions

    def predict_proba(self, dataset, multi_label_threshold: float = 0.5, dropout_sampling=1):
        """
        Parameters
        ----------
        dataset : small_text.data.datasets.Dataset
            A dataset whose labels will be predicted.
        dropout_sampling : int, default=1
            If `dropout_sampling > 1` then all dropout modules will be enabled during prediction and
            multiple rounds of predictions will be sampled for each instance.

        Returns
        -------
        scores : np.ndarray
            Distribution of confidence scores over all classes of shape (num_samples, num_classes).
            If `dropout_sampling > 1` then the shape is (num_samples, dropout_sampling, num_classes).
        """
        if len(dataset) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=False,
                                return_proba=True)

        self.model.eval()
        dataset_iter = dataloader(dataset.data, self.mini_batch_size, self._create_collate_fn(),
                                  train=False)

        logits_transform = torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)

        with inference_mode():
            with torch.autocast(device_type=self.amp_args.device_type, dtype=torch.bfloat16,
                                enabled=self.amp_args.use_amp):
                if dropout_sampling <= 1:
                    proba = self._predict_proba(len(dataset), dataset_iter, logits_transform)
                else:
                    proba = self._predict_proba_dropout_sampling(len(dataset), dataset_iter, logits_transform,
                                                                 dropout_samples=dropout_sampling)
        if self.multi_label and dropout_sampling <= 1:
            _, proba = prediction_result(proba,
                                         self.multi_label,
                                         self.num_classes,
                                         return_proba=True,
                                         multi_label_threshold=multi_label_threshold)

        return proba

    def _predict_proba(self, dataset_size, dataset_iter, logits_transform):
        raise NotImplementedError('_predict_proba() needs to be implemented')

    def _predict_proba_dropout_sampling(self, dataset_size, dataset_iter, logits_transform, dropout_samples=2):
        raise NotImplementedError('_predict_proba_dropout_sampling() needs to be implemented')

    def _get_default_criterion(self, class_weights, use_sample_weights=False):

        reduction = 'none' if use_sample_weights else 'mean'
        if self.multi_label or self.num_classes == 2:
            loss = BCEWithLogitsLoss(pos_weight=class_weights, reduction=reduction)
            if use_sample_weights:
                loss = _LossAdapter2DTo1D(loss)
            return loss
        else:
            return CrossEntropyLoss(weight=class_weights, reduction=reduction)

    def _get_default_early_stopping(self,
                                    early_stopping,
                                    validations_per_epoch,
                                    early_stopping_patience=5,
                                    early_stopping_acc=0.99):

        if early_stopping is None:
            patience = early_stopping_patience * validations_per_epoch
            early_stopping = EarlyStoppingOrCondition([
                EarlyStopping(Metric('val_loss'), patience=patience),
                EarlyStopping(Metric('train_acc', lower_is_better=False),
                              patience=-1,
                              threshold=early_stopping_acc)
            ])
        elif early_stopping == 'none':
            early_stopping = NoopEarlyStopping()

        return early_stopping

    @staticmethod
    def _get_default_model_selection(model_selection):

        if model_selection is None:
            return NoopModelSelection()

        return model_selection

    def _get_optimizer_and_scheduler(self, optimizer, scheduler, num_epochs, sub_train):

        if optimizer is None or scheduler is None:

            optimizer, scheduler = self._initialize_optimizer_and_scheduler(optimizer,
                                                                            scheduler,
                                                                            num_epochs,
                                                                            sub_train,
                                                                            self.lr)
        return optimizer, scheduler

    def _initialize_optimizer_and_scheduler(self, optimizer, scheduler, num_epochs,
                                            sub_train, base_lr):

        steps = (len(sub_train) // self.mini_batch_size) \
                + int(len(sub_train) % self.mini_batch_size != 0)

        if optimizer is None:
            params, optimizer = self._default_optimizer(base_lr) if optimizer is None else optimizer

        if scheduler == 'linear':
            try:
                from transformers import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=0,
                                                            num_training_steps=steps*num_epochs)
            except ImportError:
                raise ValueError('Linear scheduler is only available when the transformers '
                                 'integration is installed ')

        elif scheduler is None:
            # constant learning rate
            scheduler = LambdaLR(optimizer, lambda _: 1)

        return optimizer, scheduler

    def initialize_class_weights(self, sub_train):
        if self.class_weight == 'balanced':
            if self.multi_label:
                warnings.warn('Setting class_weight to \'balanced\' is intended for the '
                              'single-label use case and might not have a beneficial '
                              'effect for multi-label classification')
            class_weights_ = get_class_weights(sub_train.y, self.num_classes)
            class_weights_ = class_weights_.to(self.device)
        elif self.class_weight is None:
            class_weights_ = None
        else:
            raise ValueError(f'Invalid value for class_weight kwarg: {self.class_weight}')

        return class_weights_

    def sum_up_accuracy_(self, logits, cls):
        if self.multi_label:
            proba = torch.sigmoid(logits)
            thresholded = F.threshold(proba, 0.5, 0)
            thresholded[thresholded > 0] = 1
            num_labels = logits.shape[1]
            acc = (thresholded == cls).sum(axis=1) / num_labels
            acc = acc.sum()
        else:
            acc = (logits.argmax(1) == cls).sum()

        return acc
