import datetime
import logging
import tempfile
import warnings

import numpy as np

from functools import partial
from typing import Union

from small_text.classifiers.classification import EmbeddingMixin
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.classification import (
    _check_classifier_dataset_consistency,
    get_splits
)
from small_text.utils.context import build_pbar_context
from small_text.utils.data import check_training_data
from small_text.utils.datetime import format_timedelta

from small_text.integrations.transformers.classifiers.base import (
    ModelLoadingStrategy,
    get_default_model_loading_strategy
)

from small_text.training.model_selection import NoopModelSelection
from small_text.utils.logging import verbosity_logger, VERBOSITY_MORE_VERBOSE
from small_text.utils.system import get_show_progress_bar_default, get_tmp_dir_base

try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    from torch.optim import AdamW

    from small_text.integrations.pytorch.classifiers.base import (
        _check_optimizer_and_scheduler_config,
        PytorchClassifier
    )
    from small_text.integrations.pytorch.utils.contextmanager import inference_mode
    from small_text.integrations.pytorch.utils.data import dataloader
    from small_text.integrations.pytorch.utils.misc import _compile_if_possible, enable_dropout

    from small_text.integrations.transformers.datasets import TransformersDataset
    from small_text.integrations.transformers.utils.classification import (
        _check_for_managed_config_kwargs,
        _check_for_managed_tokenizer_kwargs,
        _check_for_managed_model_kwargs,
        _initialize_transformer_components,
        _build_layer_specific_params
    )
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


def transformers_collate_fn(batch, multi_label=None, num_classes=None, use_sample_weights=False):
    with torch.no_grad():
        text = torch.cat([entry[TransformersDataset.INDEX_TEXT] for entry in batch], dim=0)
        masks = torch.cat([entry[TransformersDataset.INDEX_MASK] for entry in batch], dim=0)
        if multi_label:
            multi_hot = [[0 if i not in set(entry[TransformersDataset.INDEX_LABEL]) else 1
                         for i in range(num_classes)]
                         for entry in batch]
            label = torch.tensor(multi_hot, dtype=float)
        else:
            label = torch.tensor([entry[TransformersDataset.INDEX_LABEL] for entry in batch])

    if use_sample_weights:
        weights = torch.tensor([entry[-1]
                                for entry in batch])
    else:
        weights = torch.ones(text.size(0), device=text.device)

    return text, masks, label, weights


class FineTuningArguments(object):
    """
    Arguments to enable and configure gradual unfreezing and discriminative learning rates as used
    in Universal Language Model Fine-tuning (ULMFiT) [HR18]_.
    """

    def __init__(self, base_lr, layerwise_gradient_decay, gradual_unfreezing=-1, cut_fraction=0.1):

        if base_lr <= 0:
            raise ValueError('FineTuningArguments: base_lr must be greater than zero')
        if layerwise_gradient_decay:
            if not (0 < layerwise_gradient_decay < 1 or layerwise_gradient_decay == -1):
                raise ValueError('FineTuningArguments: valid values for layerwise_gradient_decay '
                                 'are between 0 and 1 (or set it to -1 to disable it)')

        self.base_lr = base_lr
        self.layerwise_gradient_decay = layerwise_gradient_decay

        self.gradual_unfreezing = gradual_unfreezing
        # deprecated: This will be removed in the next version
        _unused = cut_fraction


class TransformerModelArguments(object):
    """Model arguments for :py:class:`TransformerBasedClassification`.
    """
    def __init__(self,
                 model,
                 tokenizer=None,
                 config=None,
                 model_kwargs={},
                 tokenizer_kwargs={},
                 config_kwargs={},
                 show_progress_bar : Union[None, bool] = None,
                 model_loading_strategy: ModelLoadingStrategy = get_default_model_loading_strategy(),
                 compile_model: bool = False):
        """
        Parameters
        ----------
        model : str
            Name of the transformer model. Will be passed into `AutoModel.from_pretrained()`.
        tokenizer : str, default=None
            Name of the tokenizer if deviating from the model name. Will be passed
            into `AutoTokenizer.from_pretrained()`.
        config : str, default=None
            Name of the config if deviating from the model name. Will be passed into
            `AutoConfig.from_pretrained()`.
        model_kwargs : dict, default={}
            Additional kwargs that will be passed into `AutoModelForSequenceClassification.from_pretrained()`.
            Arguments that are managed by small-text (such as the model name given by `model`) are excluded.

            .. seealso::

                `AutoModelForSequenceClassification.from_pretrained()
                <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification>`_
                in transformers.

        tokenizer_kwargs : dict, default={}
            Additional kwargs that will be passed into `AutoTokenizer.from_pretrained()`.
            Arguments that are managed by small-text (such as the tokenizer name given by `tokenizer`) are excluded.

            .. seealso::

                `AutoTokenizer.from_pretrained()
                <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer>`_ in transformers.

        config_kwargs : dict, default={}
            Additional kwargs that will be passed into `AutoConfig.from_pretrained()`.
            Arguments that are managed by small-text (such as the tokenizer name given by `tokenizer`) are excluded.

            .. seealso::

                `AutoConfig.from_pretrained()
                <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig>`_ in transformers.

        show_progress_bar : None or bool, default=None
            Determines whether progress bars are shown. If none, the small-text default is used.
        model_loading_strategy: ModelLoadingStrategy, default=ModelLoadingStrategy.DEFAULT
            Specifies if there should be attempts to download the model or if only local
            files should be used.
        compile_model : bool, default=False
            Compiles the model (using `torch.compile`) if `True` and provided that
            the PyTorch version is greater or equal to 2.0.0.

            .. versionadded:: 2.0.0

        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.model_kwargs = _check_for_managed_model_kwargs(model_kwargs)
        self.tokenizer_kwargs = _check_for_managed_tokenizer_kwargs(tokenizer_kwargs)
        self.config_kwargs = _check_for_managed_config_kwargs(config_kwargs)

        if self.tokenizer is None:
            self.tokenizer = model
        if self.config is None:
            self.config = model

        if show_progress_bar is None:
            self.show_progress_bar = get_show_progress_bar_default()
        else:
            self.show_progress_bar = show_progress_bar

        self.model_loading_strategy = model_loading_strategy
        self.compile_model = compile_model


class TransformerBasedEmbeddingMixin(EmbeddingMixin):

    EMBEDDING_METHOD_AVG = 'avg'
    EMBEDDING_METHOD_CLS_TOKEN = 'cls'

    def embed(self, data_set, return_proba=False, embedding_method=EMBEDDING_METHOD_AVG,
              hidden_layer_index=-1):
        """Embeds each sample in the given `data_set`.

        The embedding is created by using hidden representation from the transformer model's
        representation in the hidden layer at the given `hidden_layer_index`.

        Parameters
        ----------
        data_set : TransformersDataset
            The dataset for which embeddings (and class probabilities) will be computed.
        return_proba : bool
            Also return the class probabilities for `data_set`.
        embedding_method : str
            Embedding method to use ['avg', 'cls'].
        hidden_layer_index : int, default=-1
            Index of the hidden layer.

        Returns
        -------
        embeddings : np.ndarray
            Embeddings in the shape (N, hidden_layer_dimensionality).
        proba : np.ndarray
            Class probabilities in the shape (N, num_classes) for `data_set` (only if `return_predictions` is `True`).
        """

        if self.model is None:
            raise ValueError('Model is not trained. Please call fit() first.')

        self.model.eval()

        train_iter = dataloader(data_set.data, self.mini_batch_size, self._create_collate_fn(), train=False)

        tensors = []
        proba = []

        with inference_mode():
            with torch.autocast(device_type=self.amp_args.device_type, dtype=self.amp_args.dtype,
                                enabled=self.amp_args.use_amp):
                with build_pbar_context(self.transformer_model.show_progress_bar,
                                        tqdm_kwargs={'total': len(data_set)}) as pbar:
                    for batch in train_iter:
                        batch_len, logits, embeddings = self._create_embeddings(tensors,
                                                                                batch,
                                                                                embedding_method=embedding_method,
                                                                                hidden_layer_index=hidden_layer_index)
                        pbar.update(batch_len)
                        if return_proba:
                            proba.extend(F.softmax(logits, dim=1).detach().to('cpu').tolist())
                        tensors.extend(embeddings)

        if return_proba:
            return np.array(tensors), np.array(proba)

        return np.array(tensors)

    def _create_embeddings(self, tensors, batch, embedding_method='avg', hidden_layer_index=-1):

        text, masks, *_ = batch
        text = text.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)

        outputs = self.model(text,
                             attention_mask=masks,
                             output_hidden_states=True)

        # only use states of hidden layers, excluding the token embeddings
        hidden_states = outputs.hidden_states[1:]

        if embedding_method == self.EMBEDDING_METHOD_CLS_TOKEN:
            representation = hidden_states[hidden_layer_index][:, 0]
        elif embedding_method == self.EMBEDDING_METHOD_AVG:
            representation = torch.mean(hidden_states[hidden_layer_index][:, 1:], dim=1)
        else:
            raise ValueError(f'Invalid embedding_method: {embedding_method}')

        embeddings = representation.detach().to('cpu', non_blocking=True).numpy()

        return text.size(0), outputs.logits, embeddings


class TransformerBasedClassification(TransformerBasedEmbeddingMixin, PytorchClassifier):

    def __init__(self,
                 transformer_model: TransformerModelArguments,
                 num_classes: int,
                 multi_label: bool = False,
                 num_epochs: int = 10,
                 lr: float = 2e-5,
                 mini_batch_size: int = 12,
                 validation_set_size: float = 0.1,
                 validations_per_epoch: int = 1,
                 fine_tuning_arguments=None,
                 device=None,
                 memory_fix=1,
                 class_weight=None,
                 amp_args=None,
                 verbosity=VERBOSITY_MORE_VERBOSE,
                 cache_dir='.active_learning_lib_cache/'):
        """
        Parameters
        ----------
        transformer_model : TransformerModelArguments
            Settings for transformer model, tokenizer and config.
        num_classes : int
            Number of classes.
        multi_label : bool, default=False
            If `False`, the classes are mutually exclusive, i.e. the prediction step results in
            exactly one predicted label per instance.
        num_epochs : int, default=10
            Epochs to train.
        lr : float, default=2e-5
            Learning rate.
        mini_batch_size : int, default=12
            Size of mini batches during training.
        validation_set_size : float, default=0.1
            The size of the validation set as a fraction of the training set.
        validations_per_epoch : int, default=1
            Defines how of the validation set is evaluated during the training of a single epoch.
        fine_tuning_arguments : FineTuningArguments or None, default=None
            Fine-tuning arguments.
        device : str or torch.device, default=None
            Torch device on which the computation will be performed.
        memory_fix : int, default=1
            If this value is greater than zero, every `memory_fix`-many epochs the cuda cache will
            be emptied to force unused GPU memory being released.
        class_weight : 'balanced' or None, default=None
            If 'balanced', then the loss function is weighted inversely proportional to the
            label distribution to the current train set.
            label distribution to the current train set.
        amp_args : AMPArguments, default=None
            Configures the use of Automatic Mixed Precision (AMP).

            .. seealso:: :py:class:`~small_text.integrations.pytorch.classifiers.base.AMPArguments`
            .. versionadded:: 2.0.0

        verbosity : int
            Controls the verbosity of logging messages. Lower values result in less log messages.
            Set this to `VERBOSITY_QUIET` or `0` for the minimum amount of logging.
        """
        super().__init__(multi_label=multi_label, device=device, mini_batch_size=mini_batch_size,
                         amp_args=amp_args)

        with verbosity_logger():
            self.logger = logging.getLogger(__name__)
            self.logger.verbosity = verbosity

        # Training parameters
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.lr = lr

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.validation_set_size = validation_set_size
        self.validations_per_epoch = validations_per_epoch

        self.transformer_model = transformer_model

        # Other
        self.class_weight = class_weight

        self.fine_tuning_arguments = fine_tuning_arguments

        self.memory_fix = memory_fix
        self.verbosity = verbosity
        self.cache_dir = cache_dir

        self.model = None
        self.model_selection_manager = None

    def fit(self, train_set, validation_set=None, weights=None, early_stopping=None,
            model_selection=None, optimizer=None, scheduler=None):
        """Trains the model using the given train set.

        Parameters
        ----------
        train_set : TransformersDataset
            Training set.
        validation_set : TransformersDataset, default=None
            A validation set used for validation during training, or `None`. If `None`, the fit
            operation will split apart a subset of the trainset as a validation set, whose size
            is set by `self.validation_set_size`.
        weights : np.ndarray[np.float32] or None, default=None
            Sample weights or None.
        early_stopping : EarlyStoppingHandler or 'none'
            A strategy for early stopping. Passing 'none' disables early stopping.
        model_selection : ModelSelectionHandler or None, default=None
            A model selection handler. Passing 'none' disables model selection.
        optimizer : torch.optim.optimizer.Optimizer or None, default=None
            A pytorch optimizer.
        scheduler : torch.optim.LRScheduler or None, default=None
            A pytorch scheduler.

        Returns
        -------
        self : TransformerBasedClassification
            Returns the current classifier with a fitted model.
        """
        _check_classifier_dataset_consistency(self, train_set, dataset_name_in_error='training')
        _check_classifier_dataset_consistency(self, validation_set, dataset_name_in_error='validation')
        check_training_data(train_set, validation_set, weights=weights)

        optimizer_or_scheduler_given = optimizer is not None or scheduler is not None
        if self.fine_tuning_arguments is not None and optimizer_or_scheduler_given:
            raise ValueError('When fine_tuning_arguments are provided you cannot pass '
                             'optimizer and scheduler to fit()')

        if weights is not None:
            sub_train, sub_valid, sub_train_weights = get_splits(
                train_set,
                validation_set,
                weights=weights,
                multi_label=self.multi_label,
                validation_set_size=self.validation_set_size
            )
        else:
            sub_train, sub_valid = get_splits(
                train_set,
                validation_set,
                multi_label=self.multi_label,
                validation_set_size=self.validation_set_size
            )
            sub_train_weights = None

        early_stopping = self._get_default_early_stopping(early_stopping, self.validations_per_epoch)
        model_selection = self._get_default_model_selection(model_selection)

        fit_optimizer = optimizer if optimizer is not None else self.optimizer
        fit_scheduler = scheduler if scheduler is not None else self.scheduler

        self.class_weights_ = self.initialize_class_weights(sub_train)
        self.criterion = self._get_default_criterion(self.class_weights_,
                                                     use_sample_weights=weights is not None)

        return self._fit_main(sub_train, sub_valid, sub_train_weights, early_stopping,
                              model_selection, fit_optimizer, fit_scheduler)

    def _fit_main(self, sub_train, sub_valid, weights, early_stopping, model_selection,
                  optimizer, scheduler):
        if self.model is None:
            self.initialize()

        _check_optimizer_and_scheduler_config(optimizer, scheduler)
        scheduler = scheduler if scheduler is not None else 'linear'

        optimizer, scheduler = self._get_optimizer_and_scheduler(optimizer,
                                                                 scheduler,
                                                                 self.num_epochs,
                                                                 sub_train)
        self.model = self.model.to(self.device)

        with tempfile.TemporaryDirectory(dir=get_tmp_dir_base()) as tmp_dir:
            self._train(sub_train, sub_valid, weights, early_stopping, model_selection,
                        optimizer, scheduler, tmp_dir)

            if not isinstance(model_selection, NoopModelSelection):
                self._perform_model_selection(optimizer, model_selection)

        return self

    def initialize(self):
        self.config, self.tokenizer, self.model = _initialize_transformer_components(
            self.transformer_model,
            self.num_classes,
            self.cache_dir,
        )
        self.model = _compile_if_possible(self.model, self.transformer_model.compile_model)
        return self.model

    def _default_optimizer(self, base_lr):

        if self.fine_tuning_arguments is not None:
            params = _build_layer_specific_params(self.model, self.lr, self.fine_tuning_arguments)
        else:
            params = [param for param in self.model.parameters() if param.requires_grad]

        return params, AdamW(params, lr=base_lr, eps=1e-8)

    def _train(self, sub_train, sub_valid, weights, early_stopping, model_selection, optimizer,
               scheduler, tmp_dir):

        scaler = torch.amp.GradScaler(enabled=self.amp_args.use_amp)

        stop = False
        for epoch in range(0, self.num_epochs):
            if not stop:
                start_time = datetime.datetime.now()

                train_acc, train_loss, valid_acc, valid_loss, stop = self._train_loop_epoch(epoch,
                                                                                            sub_train,
                                                                                            sub_valid,
                                                                                            weights,
                                                                                            early_stopping,
                                                                                            model_selection,
                                                                                            optimizer,
                                                                                            scheduler,
                                                                                            scaler,
                                                                                            self.amp_args,
                                                                                            tmp_dir)

                timedelta = datetime.datetime.now() - start_time

                self._log_epoch(epoch, timedelta, sub_train, sub_valid, train_acc, train_loss,
                                valid_acc, valid_loss)

    def _train_loop_epoch(self, num_epoch, sub_train, sub_valid, weights, early_stopping,
                          model_selection, optimizer, scheduler, scaler, amp_args, tmp_dir):

        if self.memory_fix and (num_epoch + 1) % self.memory_fix == 0:
            torch.cuda.empty_cache()

        self.model.train()
        if self.validations_per_epoch > 1:
            num_batches = len(sub_train) // self.mini_batch_size \
                          + int(len(sub_train) % self.mini_batch_size > 0)
            if self.validations_per_epoch > num_batches:
                warnings.warn(
                    f'validations_per_epoch={self.validations_per_epoch} is greater than '
                    f'the maximum possible batches of {num_batches}',
                    RuntimeWarning)
                validate_every = 1
            else:
                validate_every = int(num_batches / self.validations_per_epoch)
        else:
            validate_every = None

        train_loss, train_acc, valid_loss, valid_acc, stop = self._train_loop_process_batches(
            num_epoch,
            sub_train,
            sub_valid,
            weights,
            early_stopping,
            model_selection,
            optimizer,
            scheduler,
            scaler,
            tmp_dir,
            validate_every=validate_every)

        return train_acc, train_loss, valid_acc, valid_loss, stop

    def _train_loop_process_batches(self, num_epoch, sub_train_, sub_valid_, weights, early_stopping,
                                    model_selection, optimizer, scheduler, scaler, tmp_dir, validate_every=None):

        train_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        train_acc = torch.tensor(0., dtype=torch.float32, device=self.device)
        valid_losses = []
        valid_accs = []

        data = sub_train_.data
        if weights is not None:
            data = [d + (weights[i],) for i, d in enumerate(data)]

        train_iter = dataloader(data, self.mini_batch_size,
                                self._create_collate_fn(use_sample_weights=weights is not None))

        stop = False

        for i, (x, masks, cls, weight, *_) in enumerate(train_iter):
            if not stop:
                with torch.autocast(enabled=self.amp_args.use_amp, device_type=self.amp_args.device_type,
                                    dtype=self.amp_args.dtype):
                    loss, logits, cls = self._train_forward(x, masks, cls, weight, optimizer)
                del x, masks
                self._train_backward(loss, optimizer, scaler)

                scheduler.step()

                train_loss += loss.detach()
                train_acc += self.sum_up_accuracy_(logits, cls).detach()
                del cls

                if validate_every and i % validate_every == 0:
                    valid_loss, valid_acc = self.validate(sub_valid_)
                    self.model.train()
                    valid_losses.append(valid_loss)
                    valid_accs.append(valid_acc)

                    measured_values = {'val_loss': valid_loss, 'val_acc': valid_acc}
                    stop = stop or early_stopping.check_early_stop(num_epoch+1, measured_values)
                    if not isinstance(model_selection, NoopModelSelection):
                        self._save_model(optimizer, model_selection, f'{num_epoch}-b{i+1}',
                                         train_acc, train_loss, valid_acc, valid_loss, stop, tmp_dir)

        if validate_every:
            valid_loss, valid_acc = np.mean(valid_losses), np.mean(valid_accs)

        else:
            valid_loss, valid_acc = None, None
            if validate_every is None and sub_valid_ is not None:
                valid_loss, valid_acc = self.validate(sub_valid_)
        train_loss = train_loss.item() / len(sub_train_)
        train_acc = train_acc.item() / len(sub_train_)

        measured_values = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': valid_loss,
            'val_acc': valid_acc
        }
        stop = early_stopping.check_early_stop(num_epoch+1, measured_values)
        if not isinstance(model_selection, NoopModelSelection):
            self._save_model(optimizer, model_selection, f'{num_epoch}-b0',
                             train_acc, train_loss, valid_acc, valid_loss, stop, tmp_dir)

        return train_loss, train_acc, valid_loss, valid_acc, stop

    def _create_collate_fn(self, use_sample_weights=False):
        return partial(transformers_collate_fn, multi_label=self.multi_label,
                       num_classes=self.num_classes, use_sample_weights=use_sample_weights)

    def _train_forward(self, x, masks, cls, weight, optimizer):

        optimizer.zero_grad()

        x, masks, cls = x.to(self.device), masks.to(self.device), cls.to(self.device)
        weight = weight.to(self.device, non_blocking=True)

        outputs = self.model(x, attention_mask=masks)

        logits, loss = self._compute_loss(cls, outputs)
        loss = loss * weight
        loss = loss.mean()

        del outputs

        return loss, logits, cls

    def _train_backward(self, loss, optimizer, scaler):
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

    def _compute_loss(self, cls, outputs):
        if self.num_classes == 2:
            logits = outputs.logits
            target = F.one_hot(cls, 2).float()
        else:
            logits = outputs.logits.view(-1, self.num_classes)
            target = cls
        loss = self.criterion(logits, target)

        return logits, loss

    def _log_epoch(self, epoch, timedelta, sub_train, sub_valid, train_acc, train_loss, valid_acc, valid_loss):
        if sub_valid is not None:
            valid_loss_txt = f'\n\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)'
        else:
            valid_loss_txt = ''
        self.logger.info(f'Epoch: {epoch + 1} | {format_timedelta(timedelta)}\n'
                         f'\tTrain Set Size: {len(sub_train)}\n'
                         f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)'
                         f'{valid_loss_txt}',
                         verbosity=VERBOSITY_MORE_VERBOSE)

    def validate(self, validation_set):

        with inference_mode():
            with torch.autocast(device_type=self.amp_args.device_type, dtype=self.amp_args.dtype,
                                enabled=self.amp_args.use_amp):
                valid_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
                acc = torch.tensor(0., dtype=torch.float32, device=self.device)

                self.model.eval()
                valid_iter = dataloader(validation_set.data, self.mini_batch_size,
                                        self._create_collate_fn(),
                                        train=False)

                for x, masks, cls, weight, *_ in valid_iter:
                    x, masks, cls = x.to(self.device), masks.to(self.device), cls.to(self.device)
                    weight = weight.to(self.device, non_blocking=True)

                    outputs = self.model(x, attention_mask=masks)
                    _, loss = self._compute_loss(cls, outputs)
                    loss = loss * weight
                    loss = loss.mean()

                    valid_loss += loss.detach()
                    acc += self.sum_up_accuracy_(outputs.logits, cls).detach()
                    del outputs, x, masks, cls

        return valid_loss.item() / len(validation_set), acc.item() / len(validation_set)

    def predict(self, dataset, return_proba=False):
        """Predicts the labels for the given dataset.

        Parameters
        ----------
        dataset : TransformersDataset
            A dataset on whose instances predictions are made.
        return_proba : bool, default=False
            If True, additionally returns the confidence distribution over all classes.

        Returns
        -------
        predictions : np.ndarray[np.int32] or csr_matrix[np.int32]
            List of predictions if the classifier was fitted on single-label data,
            otherwise a sparse matrix of predictions.
        probas : np.ndarray[np.float32], optional
            List of probabilities (or confidence estimates) if `return_proba` is True.
        """
        return super().predict(dataset, return_proba=return_proba)

    def predict_proba(self, dataset, dropout_sampling=1):
        """Predicts the label distributions.

        Parameters
        ----------
        dataset : TransformersDataset
            A dataset whose labels will be predicted.
        dropout_sampling : int, default=1
            If `dropout_sampling > 1` then all dropout modules will be enabled during prediction and
            multiple rounds of predictions will be sampled for each instance.

        Returns
        -------
        scores : np.ndarray
            Confidence score distribution over all classes of shape (num_samples, num_classes).
            If `dropout_sampling > 1` then the shape is (num_samples, dropout_sampling, num_classes).
        """
        return super().predict_proba(dataset, dropout_sampling=dropout_sampling)

    def _predict_proba(self, dataset_size, dataset_iter, logits_transform):
        predictions = np.empty((dataset_size, self.num_classes), dtype=float)
        offset = 0

        for text, masks, *_ in dataset_iter:
            text, masks = text.to(self.device), masks.to(self.device)
            batch_size = text.shape[0]
            outputs = self.model(text, attention_mask=masks)

            predictions[offset:offset+batch_size] = logits_transform(outputs.logits).to('cpu').numpy()

            offset += batch_size
            del text, masks
        return predictions

    def _predict_proba_dropout_sampling(self, dataset_size, dataset_iter, logits_transform, dropout_samples=2):

        predictions = np.empty((dataset_size, dropout_samples, self.num_classes), dtype=float)
        offset = 0

        with enable_dropout(self.model):
            for text, masks, *_ in dataset_iter:
                batch_size, vector_len = text.shape
                full_size = batch_size * dropout_samples
                text, masks = text.to(self.device), masks.to(self.device)

                text, masks = text.repeat(1, dropout_samples).resize(full_size, vector_len), \
                    masks.repeat(1, dropout_samples).resize(full_size, vector_len)

                outputs = self.model(text, attention_mask=masks)

                prediction_for_batch = logits_transform(outputs.logits)
                prediction_for_batch = prediction_for_batch.unsqueeze(dim=1)\
                    .resize(batch_size, dropout_samples, self.num_classes)

                predictions[offset:offset+batch_size] = prediction_for_batch.to('cpu').numpy()

                offset += batch_size
                del text, masks

        return predictions

    def __del__(self):
        try:
            attrs = ['criterion', 'optimizer', 'scheduler', 'model', 'tokenizer', 'config']
            for attr in attrs:
                delattr(self, attr)
        except Exception:
            pass
