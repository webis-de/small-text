import logging
import warnings

from abc import abstractmethod
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

try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
    from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

    from small_text.integrations.pytorch.utils.data import get_class_weights
    from small_text.utils.classification import empty_result, prediction_result
    from small_text.integrations.pytorch.utils.loss import _LossAdapter2DTo1D
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


logger = logging.getLogger(__name__)


def _check_optimizer_and_scheduler_config(optimizer, scheduler):
    if scheduler is not None and optimizer is None:
        raise ValueError('You must also pass an optimizer if you pass a scheduler to fit()')


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


class PytorchClassifier(PytorchModelSelectionMixin, Classifier):

    def __init__(self, multi_label=False, device=None, mini_batch_size=32):

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

    @abstractmethod
    def fit(self, train_set, validation_set=None, weights=None, **kwargs):
        pass

    def predict(self, data_set, return_proba=False):
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
        predictions = prediction_result(proba, self.multi_label, self.num_classes)

        if return_proba:
            return predictions, proba

        return predictions

    @abstractmethod
    def predict_proba(self, test_set):
        """
        Parameters
        ----------
        test_set : small_text.integrations.pytorch.PytorchTextClassificationDataset
            Test set.

        Returns
        -------
        scores : np.ndarray
            Distribution of confidence scores over all classes.
        """
        pass

    def _get_default_criterion(self, class_weights, use_sample_weights=False):

        reduction = 'none' if use_sample_weights else 'mean'
        if self.multi_label or self.num_classes == 2:
            loss = BCEWithLogitsLoss(pos_weight=class_weights, reduction=reduction)
            if use_sample_weights:
                loss = _LossAdapter2DTo1D(loss)
            return loss
        else:
            return CrossEntropyLoss(weight=class_weights, reduction=reduction)

    def _get_default_early_stopping(self, early_stopping, early_stopping_no_improvement,
                                    early_stopping_acc, validations_per_epoch,
                                    kwarg_no_improvement_name='early_stopping_no_improvement'):

        if early_stopping is None:
            patience = early_stopping_no_improvement * validations_per_epoch
            if early_stopping_no_improvement == 5 and early_stopping_acc == -1:
                early_stopping = EarlyStopping(Metric('val_loss'), patience=patience)
            elif early_stopping_no_improvement == 0 and early_stopping_acc == -1:
                early_stopping = NoopEarlyStopping()
            else:
                early_stopping = EarlyStoppingOrCondition([
                    EarlyStopping(Metric('val_loss'), patience=patience),
                    EarlyStopping(Metric('train_acc', lower_is_better=False),
                                  patience=-1,
                                  threshold=early_stopping_acc)
                ])
        elif early_stopping == 'none':
            early_stopping = NoopEarlyStopping()
        else:
            if early_stopping_no_improvement != 5 or early_stopping_acc != -1:
                warnings.warn(f'Both the fit() argument early_stopping and the __init__() '
                              f'arguments "{kwarg_no_improvement_name}" / "early_stopping_acc" '
                              f'have been used. In this case the fit() argument takes precedence. '
                              f'The __init__() arguments are deprecated, so please use the '
                              f'early stopping argument in fit() instead.',
                              UserWarning)
        return early_stopping

    @staticmethod
    def _get_default_model_selection(model_selection):
        if model_selection == 'none':
            return NoopModelSelection()

        return ModelSelection()

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
            params, optimizer = self._default_optimizer(base_lr) \
                if optimizer is None else optimizer

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
        elif not isinstance(scheduler, _LRScheduler):
            raise ValueError(f'Invalid scheduler: {scheduler}')

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
            acc = acc.sum().item()
        else:
            acc = (logits.argmax(1) == cls).sum().item()

        return acc
