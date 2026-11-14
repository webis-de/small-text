import datetime
import logging
import tempfile

from functools import partial

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

from small_text.classifiers.classification import EmbeddingMixin
from small_text.data.splits import get_splits
from small_text.integrations.pytorch.classifiers.base import PytorchClassifier
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.integrations.pytorch.models.lstm import LSTM
from small_text.training.model_selection import NoopModelSelection
from small_text.utils.classification import (
    _check_classifier_dataset_consistency,
    empty_result,
    prediction_result
)
from small_text.utils.context import build_pbar_context
from small_text.utils.data import check_training_data, list_length
from small_text.utils.labels import csr_to_list
from small_text.utils.datetime import format_timedelta
from small_text.utils.logging import verbosity_logger, VERBOSITY_MORE_VERBOSE
from small_text.utils.system import get_show_progress_bar_default, get_tmp_dir_base

try:
    import torch
    import torch.nn.functional as F
    from torch.optim import Adadelta

    from small_text.integrations.pytorch.classifiers.base import (
        _check_optimizer_and_scheduler_config
    )
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset

    from small_text.integrations.pytorch.utils.data import dataloader
    from small_text.integrations.pytorch.utils.misc import _compile_if_possible, enable_dropout
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


def lstm_collate_fn(batch, multi_label=None, num_classes=None, use_sample_weights=False,
                    max_seq_len=60, padding_idx=0):
    def prepare_tensor(t):
        t_sub = t[:max_seq_len]
        return torch.cat([t_sub,
                          t_sub.new_zeros(max_seq_len - t_sub.size(0)) + padding_idx],
                         0)

    if multi_label:
        multi_hot = [[0 if i not in set(entry[PytorchTextClassificationDataset.INDEX_LABEL]) else 1
                      for i in range(num_classes)]
                     for entry in batch]
        label = torch.tensor(multi_hot, dtype=torch.float32)
    else:
        label = torch.tensor([entry[PytorchTextClassificationDataset.INDEX_LABEL]
                              for entry in batch])
    text = torch.stack([prepare_tensor(entry[PytorchTextClassificationDataset.INDEX_TEXT])
                        for entry in batch], 0)

    if use_sample_weights:
        weights = torch.tensor([entry[-1]
                                for entry in batch])
    else:
        weights = torch.ones(text.size(0), device=text.device)

    return text, label, weights


class LSTMEmbeddingMixin(EmbeddingMixin):

    EMBEDDING_METHOD_POOLED = 'pooled'
    EMBEDDING_METHOD_GRADIENT = 'gradient'

    def embed(self, data_set,
              return_proba: bool = False,
              multi_label_threshold: float = 0.5,
              embedding_method: str = EMBEDDING_METHOD_POOLED,
              module_selector=lambda x: x['fc']):
        """Embeds each sample in the given ``data_set``.

        The embedding is created from the final hidden state of the underlying LSTM model.

        Parameters
        ----------
        data_set : PytorchTextClassificationDataset
            The dataset for which embeddings (and class probabilities) will be computed.
        return_proba : bool, default=False
            Whether to also return class probabilities for ``data_set``.
        multi_label_threshold : float, default=0.5
            In multi-label classification, a label is predicted only when its probability is
            greater than ``multi_label_threshold``. Ignored for single-label classification.
        embedding_method : {'pooled', 'gradient'}, default='pooled'
            Method used to create embeddings. Gradient embeddings are supported only for
            single-label classification.
        module_selector : callable, default=lambda x: x['fc']
            Selects the module whose weights are used to compute gradient embeddings.

        Returns
        -------
        embeddings : np.ndarray[np.float32]
            Embeddings with shape ``(N, hidden_layer_dimensionality)``.
        proba : np.ndarray or scipy.sparse.csr_matrix
            Class probabilities with shape ``(N, num_classes)`` when ``return_proba`` is True.
        """
        if self.model is None:
            raise ValueError('Model is not trained. Please call fit() first.')
        if embedding_method == self.EMBEDDING_METHOD_GRADIENT and self.multi_label:
            raise ValueError('Gradient embeddings are only supported for single-label classification.')

        self.model.eval()

        dataset_iter = dataloader(data_set.data,
                                  self.predict_batch_size,
                                  self._create_collate_fn(),
                                  train=False)

        tensors = []
        proba = []

        requires_grad = embedding_method == self.EMBEDDING_METHOD_GRADIENT
        logits_transform = torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)
        with torch.set_grad_enabled(requires_grad):
            with build_pbar_context(self.show_progress_bar, tqdm_kwargs={'total': list_length(data_set)}) as pbar:
                for batch in dataset_iter:
                    with torch.autocast(device_type=self.amp_args.device_type, dtype=self.amp_args.dtype,
                                        enabled=self.amp_args.use_amp):
                        batch_len, logits, embeddings = self._create_embeddings(batch,
                                                                                embedding_method=embedding_method,
                                                                                module_selector=module_selector)

                    embeddings = embeddings.detach().to('cpu').numpy(force=True)

                    pbar.update(batch_len)
                    if return_proba:
                        proba.extend(logits_transform(logits).detach().to('cpu').tolist())
                    tensors.extend(embeddings)
        if return_proba:
            _, proba = prediction_result(np.array(proba),
                                         self.multi_label,
                                         self.num_classes,
                                         return_proba=return_proba,
                                         multi_label_threshold=multi_label_threshold)
            return np.array(tensors, dtype=np.float32), proba

        return np.array(tensors, dtype=np.float32)

    def _create_embeddings(self, batch, embedding_method='pooled', module_selector=lambda x: x['fc']):
        text, *_ = batch

        text = text.to(self.device)

        if embedding_method == self.EMBEDDING_METHOD_POOLED:
            embeddings = self.model._forward_pooled(text)
            logits = self.model._dropout_and_fc(embeddings)
        elif embedding_method == self.EMBEDDING_METHOD_GRADIENT:
            best_label, logits = self._get_best_and_softmax(text)
            embeddings = self._create_gradient_embedding(best_label, logits, module_selector, text)
        else:
            raise ValueError(f'Invalid embedding method: {embedding_method}')

        if self.amp_args.use_amp:
            embeddings = embeddings.float()
            logits = logits.float()

        return text.size(0), logits, embeddings

    def _get_best_and_softmax(self, text):

        self.model.zero_grad()

        logits = self.model(text)

        sm = F.softmax(logits, dim=1)
        with torch.no_grad():
            best_label = torch.argmax(sm, dim=1)

        return best_label, logits

    def _create_gradient_embedding(self, best_label, logits, module_selector, text):
        # use logits-based cross-entropy loss with explicit targets
        batch_len = text.size(0)
        probabilities = F.softmax(logits, dim=1)

        modules = dict({name: module for name, module in self.model.named_modules()})
        grad_size = module_selector(modules).weight.flatten().size(0)
        embeddings = torch.empty(batch_len, grad_size * self.num_classes,
                                 dtype=torch.float32, device=logits.device)

        for c in range(self.num_classes):
            target = torch.full((batch_len,), c, dtype=torch.long, device=self.device)
            if self.num_classes == 2:
                target = F.one_hot(target, self.num_classes).to(dtype=logits.dtype)
                loss = F.binary_cross_entropy_with_logits(logits, target,
                                                          pos_weight=self.class_weights_,
                                                          reduction='none').mean(dim=1)
            else:
                loss = F.cross_entropy(logits, target, weight=self.class_weights_, reduction='none')

            with torch.autocast(device_type=self.amp_args.device_type, dtype=self.amp_args.dtype,
                                enabled=False):
                for k in range(batch_len):
                    self.model.zero_grad()
                    loss[k].backward(retain_graph=True)

                    modules = dict({name: module for name, module in self.model.named_modules()})
                    gradients = module_selector(modules).weight.grad.flatten()

                    with torch.no_grad():
                        probability = probabilities[k, c]
                        if c == best_label[k].item():
                            embeddings[k, grad_size*c:grad_size*(c+1)] = \
                                (1-probability) * gradients
                        else:
                            embeddings[k, grad_size*c:grad_size*(c+1)] = -probability * gradients

        return embeddings


class LSTMClassifier(LSTMEmbeddingMixin, PytorchClassifier):

    def __init__(self,
                 num_classes,
                 multi_label=False,
                 embedding_matrix=None,
                 device=None,
                 num_epochs: int = 10,
                 train_batch_size: int = 25,
                 predict_batch_size: int = 25,
                 lr=0.001,
                 max_seq_len=60,
                 hidden_size=100,
                 dropout=0.5,
                 validation_set_size=0.1,
                 padding_idx=0,
                 bidirectional=False,
                 freeze_embedding_layer=False,
                 show_progress_bar=None,
                 class_weight=None,
                 amp_args=None,
                 compile_model=False,
                 verbosity=VERBOSITY_MORE_VERBOSE):
        """Initializes an LSTM classifier.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        multi_label : bool, default=False
            If ``False``, classes are mutually exclusive and prediction returns one label per
            instance.
        embedding_matrix : torch.FloatTensor
            Embeddings with shape ``(vocab_size, embedding_size)``.
        device : str or torch.device, default=None
            Torch device on which computation is performed.
        num_epochs : int, default=10
            Number of training epochs.
        train_batch_size : int, default=25
            Batch size during training.
        predict_batch_size : int, default=25
            Batch size during prediction.
        lr : float, default=0.001
            Learning rate.
        max_seq_len : int, default=60
            Maximum sequence length.
        hidden_size : int, default=100
            Number of features in the LSTM hidden state.
        dropout : float, default=0.5
            Dropout probability applied before the classification layer.
        validation_set_size : float, default=0.1
            Fraction of the training set used as validation data if no validation set is given.
        padding_idx : int, default=0
            Index of the padding token in the vocabulary.
        bidirectional : bool, default=False
            Whether to use a bidirectional LSTM.
        freeze_embedding_layer : bool, default=False
            Whether to keep the supplied embedding matrix fixed during training.
        show_progress_bar : bool or None, default=None
            Whether to display progress bars. ``None`` uses the small-text default.
        class_weight : {'balanced', None}, default=None
            If ``'balanced'``, weight the loss inversely to label frequencies in the training
            data.
        amp_args : AMPArguments, default=None
            Configuration for Automatic Mixed Precision (AMP).

            .. seealso:: :py:class:`~small_text.integrations.pytorch.classifiers.base.AMPArguments`
        compile_model : bool, default=False
            Compiles the model with ``torch.compile`` when supported by the installed PyTorch
            version.
        verbosity : int, default=VERBOSITY_MORE_VERBOSE
            Verbosity level used for training logs.
        """
        super().__init__(multi_label=multi_label,
                         device=device,
                         train_batch_size=train_batch_size,
                         predict_batch_size=predict_batch_size,
                         amp_args=amp_args)

        with verbosity_logger():
            self.logger = logging.getLogger(__name__)
            self.logger.verbosity = verbosity

        if embedding_matrix is None:
            raise ValueError('This implementation requires an embedding matrix.')

        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.lr = lr

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.class_weight = class_weight
        if show_progress_bar is None:
            self.show_progress_bar = get_show_progress_bar_default()
        else:
            self.show_progress_bar = show_progress_bar
        self.compile_model = compile_model

        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.validation_set_size = validation_set_size
        self.embedding_matrix = embedding_matrix
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.freeze_embedding_layer = freeze_embedding_layer

        self.model = None
        self.model_selection = None
        self.enc_ = None

    def fit(self, train_set, validation_set=None, weights=None, early_stopping=None,
            model_selection=None, optimizer=None, scheduler=None):
        """Trains the classifier using the given training set.

        Parameters
        ----------
        train_set : PytorchTextClassificationDataset
            Dataset used for training.
        validation_set : PytorchTextClassificationDataset or None, default=None
            Validation data. If omitted, a validation subset is split from ``train_set``.
        weights : np.ndarray[np.float32] or None, default=None
            Per-sample training weights.
        early_stopping : EarlyStoppingHandler or {'none'}, default=None
            Early-stopping strategy. ``'none'`` disables early stopping.
        model_selection : ModelSelectionHandler or {'none'}, default=None
            Model-selection strategy. ``'none'`` disables model selection.
        optimizer : torch.optim.Optimizer or None, default=None
            Optimizer to use for training.
        scheduler : torch.optim.lr_scheduler.LRScheduler or {'linear'} or None, default=None
            Learning-rate scheduler to use for training.

        Returns
        -------
        self : LSTMClassifier
            The fitted classifier.
        """
        _check_classifier_dataset_consistency(self, train_set, dataset_name_in_error='training')
        _check_classifier_dataset_consistency(self, validation_set, dataset_name_in_error='validation')
        check_training_data(train_set, validation_set, weights=weights)

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

        early_stopping = self._get_default_early_stopping(early_stopping, 1)
        model_selection = self._get_default_model_selection(model_selection)

        fit_optimizer = optimizer if optimizer is not None else self.optimizer
        fit_scheduler = scheduler if scheduler is not None else self.scheduler

        if self.multi_label:
            self.enc_ = MultiLabelBinarizer()
            labels = csr_to_list(sub_train.y)
            self.enc_ = self.enc_.fit(labels)

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
        scheduler = scheduler if scheduler is not None else None

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
        vocab_size = self.embedding_matrix.shape[0]
        embed_dim = self.embedding_matrix.shape[1]
        self.model = LSTM(vocab_size, self.max_seq_len, num_classes=self.num_classes,
                          hidden_size=self.hidden_size, dropout=self.dropout,
                          embedding_matrix=self.embedding_matrix, embed_dim=embed_dim,
                          freeze_embedding_layer=self.freeze_embedding_layer,
                          padding_idx=self.padding_idx, bidirectional=self.bidirectional)

        self.model = _compile_if_possible(self.model, compile_model=self.compile_model)
        return self.model

    def _default_optimizer(self, base_lr):
        params = [param for param in self.model.parameters() if param.requires_grad]
        return params, Adadelta(params, lr=base_lr, eps=1e-8)

    def _train(self, sub_train, sub_valid, weights, early_stopping, model_selection, optimizer,
               scheduler, tmp_dir):
        scaler = torch.amp.GradScaler(enabled=self.amp_args.use_amp)

        stop = False
        for epoch in range(self.num_epochs):
            if not stop:
                start_time = datetime.datetime.now()

                self.model.train()
                train_loss, train_acc = self._train_func(sub_train,
                                                         weights,
                                                         optimizer,
                                                         scheduler,
                                                         scaler,
                                                         self.amp_args)

                self.model.eval()
                valid_loss, valid_acc = self.validate(sub_valid)

                timedelta = datetime.datetime.now() - start_time

                self.logger.info(f'Epoch: {epoch + 1} | {format_timedelta(timedelta)}\n'
                                 f'\tTrain Set Size: {len(sub_train)}\n'
                                 f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}% (train)\n'
                                 f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}% (valid)',
                                 verbosity=VERBOSITY_MORE_VERBOSE)

                measured_values = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': valid_loss,
                    'val_acc': valid_acc
                }
                stop = early_stopping.check_early_stop(epoch + 1, measured_values)
                if not isinstance(model_selection, NoopModelSelection):
                    self._save_model(optimizer, model_selection, f'{epoch}-0',
                                     train_acc, train_loss, valid_acc, valid_loss, stop, tmp_dir)

        return self

    def _create_collate_fn(self, use_sample_weights=False):
        return partial(lstm_collate_fn, multi_label=self.multi_label,
                       num_classes=self.num_classes, use_sample_weights=use_sample_weights,
                       padding_idx=self.padding_idx, max_seq_len=self.max_seq_len)

    def _train_func(self, sub_train_, weights, optimizer, scheduler, scaler, amp_args):

        train_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        train_acc = torch.tensor(0., dtype=torch.float32, device=self.device)

        data = sub_train_.data
        if weights is not None:
            data = [d + (weights[i],) for i, d in enumerate(data)]

        train_iter = dataloader(data, self.train_batch_size,
                                self._create_collate_fn(use_sample_weights=weights is not None))

        for text, cls, weight in train_iter:
            with torch.autocast(enabled=amp_args.use_amp, device_type=amp_args.device_type,
                                dtype=amp_args.dtype):
                loss, acc = self._train_single_batch(text, cls, weight, optimizer, scaler)
                scheduler.step()

                train_loss += loss.detach() * text.size(0)
                train_acc += acc.detach()

        return train_loss.item() / len(sub_train_), train_acc.item() / len(sub_train_)

    def _train_single_batch(self, text, cls, weight, optimizer, scaler):

        optimizer.zero_grad()

        text, cls, weight = text.to(self.device), cls.to(self.device), weight.to(self.device)
        output = self.model(text)

        loss = self._compute_loss(cls, output)
        loss = loss * weight
        loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)

        scaler.step(optimizer)
        scaler.update()

        train_acc = self.sum_up_accuracy_(output, cls)

        del text, cls, output

        return loss, train_acc

    def _compute_loss(self, cls, output):
        with torch.no_grad():
            if self.multi_label:
                target = cls.float()
            elif self.num_classes == 2:
                target = F.one_hot(cls, 2).float()
            else:
                target = cls
        loss = self.criterion(output, target)
        return loss

    def validate(self, validation_set):
        """Obtains validation loss and accuracy for ``validation_set``.

        Parameters
        ----------
        validation_set : PytorchTextClassificationDataset
            Dataset on which to evaluate the classifier.

        Returns
        -------
        validation_loss : float
            Mean loss over ``validation_set``.
        validation_acc : float
            Mean accuracy over ``validation_set``.
        """

        with torch.no_grad():
            with torch.autocast(device_type=self.amp_args.device_type, dtype=self.amp_args.dtype,
                                enabled=self.amp_args.use_amp):

                valid_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
                acc = torch.tensor(0., dtype=torch.float32, device=self.device)

                self.model.eval()
                valid_iter = dataloader(validation_set.data, self.predict_batch_size, self._create_collate_fn(),
                                        train=False)

                for x, cls, weight, *_ in valid_iter:
                    x, cls, weight = x.to(self.device), cls.to(self.device), weight.to(self.device)

                    with torch.no_grad():
                        output = self.model(x)
                        loss = self._compute_loss(cls, output)
                        loss = loss * weight
                        loss = loss.mean()

                        valid_loss += loss.detach() * x.size(0)

                        acc += self.sum_up_accuracy_(output, cls)
                        del output, x, cls

                return valid_loss.item() / len(validation_set), acc.item() / len(validation_set)

    def predict(self, dataset, return_proba=False, multi_label_threshold: float = 0.5):
        """Predicts labels for the given dataset.

        Parameters
        ----------
        dataset : PytorchTextClassificationDataset
            Dataset whose instances are predicted.
        return_proba : bool, default=False
            Whether to also return class probabilities.
        multi_label_threshold : float, default=0.5
            In multi-label classification, predict a label only when its probability is greater
            than this threshold. Ignored for single-label classification.

        Returns
        -------
        predictions : np.ndarray[np.int64] or scipy.sparse.csr_matrix
            Predicted labels.
        proba : np.ndarray or scipy.sparse.csr_matrix
            Class probabilities when ``return_proba`` is True.
        """
        if len(dataset) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=True,
                                return_proba=return_proba)

        proba = self.predict_proba(dataset, multi_label_threshold=multi_label_threshold)
        predictions = prediction_result(proba.toarray() if self.multi_label else proba,
                                        self.multi_label,
                                        self.num_classes,
                                        multi_label_threshold=multi_label_threshold)

        if return_proba:
            return predictions, proba

        return predictions

    def predict_proba(self, dataset, multi_label_threshold: float = 0.5, dropout_sampling=1):
        """Predicts class probabilities for the given dataset.

        Parameters
        ----------
        dataset : PytorchTextClassificationDataset
            Dataset whose instances are predicted.
        multi_label_threshold : float, default=0.5
            In multi-label classification, retain probabilities only for labels above this
            threshold. Ignored for single-label classification.
        dropout_sampling : int, default=1
            If greater than one, enable dropout during prediction and draw this many samples per
            instance.

        Returns
        -------
        scores : np.ndarray or scipy.sparse.csr_matrix
            Class probabilities with shape ``(num_samples, num_classes)``. With dropout
            sampling, the dense result has shape ``(num_samples, dropout_sampling, num_classes)``.
        """
        return super().predict_proba(dataset,
                                     multi_label_threshold=multi_label_threshold,
                                     dropout_sampling=dropout_sampling)

    def _predict_proba(self, dataset_size, dataset_iter, logits_transform):
        predictions = np.empty((dataset_size, self.num_classes), dtype=float)
        offset = 0

        for text, *_ in dataset_iter:
            batch_size = text.shape[0]

            text = text.to(self.device)
            output = self.model.forward(text)

            predictions[offset:offset+batch_size] = logits_transform(output).to('cpu').numpy()

            offset += batch_size
            del text
        return predictions

    def _predict_proba_dropout_sampling(self, dataset_size, dataset_iter, logits_transform, dropout_samples=2):

        predictions = np.empty((dataset_size, dropout_samples, self.num_classes), dtype=float)
        offset = 0

        with enable_dropout(self.model):
            for text, *_ in dataset_iter:
                batch_size = text.shape[0]
                text = text.to(self.device)
                text = text.repeat_interleave(dropout_samples, dim=0)

                output = self.model.forward(text)

                prediction_for_batch = logits_transform(output).reshape(
                    batch_size, dropout_samples, self.num_classes
                )

                predictions[offset:offset+batch_size] = prediction_for_batch.to('cpu').numpy()

                offset += batch_size
                del text

        return predictions

    def __del__(self):
        try:
            attrs = ['criterion', 'optimizer', 'scheduler', 'model']
            for attr in attrs:
                delattr(self, attr)
        except Exception:
            pass
