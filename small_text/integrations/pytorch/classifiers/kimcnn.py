import datetime
import logging
import tempfile

import numpy as np

from functools import partial

from sklearn.preprocessing import MultiLabelBinarizer

from small_text.classifiers.classification import EmbeddingMixin
from small_text.integrations.pytorch.classifiers.base import PytorchClassifier
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.integrations.pytorch.models.kimcnn import KimCNN
from small_text.training.model_selection import NoopModelSelection
from small_text.utils.classification import (
    _check_classifier_dataset_consistency,
    get_splits
)
from small_text.utils.context import build_pbar_context
from small_text.utils.data import check_training_data, list_length
from small_text.utils.labels import csr_to_list
from small_text.utils.datetime import format_timedelta
from small_text.utils.logging import verbosity_logger, VERBOSITY_MORE_VERBOSE
from small_text.utils.system import get_show_progress_bar_default, get_tmp_dir_base

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from torch.optim import Adadelta

    from small_text.integrations.pytorch.classifiers.base import (
        _check_optimizer_and_scheduler_config
    )
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset

    from small_text.integrations.pytorch.utils.contextmanager import inference_mode
    from small_text.integrations.pytorch.utils.data import dataloader
    from small_text.integrations.pytorch.utils.misc import _compile_if_possible, enable_dropout
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


def kimcnn_collate_fn(batch, multi_label=None, num_classes=None, use_sample_weights=False,
                      max_seq_len=60, padding_idx=0, filter_padding=0):
    # TODO: torch.no_grad()?
    def prepare_tensor(t):
        t_sub = t[:max_seq_len-2*filter_padding]
        return torch.cat([t_sub.new_zeros(filter_padding) + padding_idx,
                          t_sub,
                          t_sub.new_zeros(max_seq_len - 2*filter_padding - t_sub.size(0)) + padding_idx,
                          t_sub.new_zeros(filter_padding) + padding_idx],
                         0)

    if multi_label:
        multi_hot = [[0 if i not in set(entry[PytorchTextClassificationDataset.INDEX_LABEL]) else 1
                      for i in range(num_classes)]
                     for entry in batch]
        label = torch.tensor(multi_hot, dtype=float)
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


class KimCNNEmbeddingMixin(EmbeddingMixin):

    EMBEDDING_METHOD_POOLED = 'pooled'
    EMBEDDING_METHOD_GRADIENT = 'gradient'

    def embed(self, data_set, return_proba=False, embedding_method=EMBEDDING_METHOD_POOLED,
              module_selector=lambda x: x['fc']):
        """Embeds each sample in the given `data_set`.

        The embedding is created by using the underlying sentence transformer model.

        Parameters
        ----------
        data_set : PytorchTextClassificationDataset
            The dataset for which embeddings (and class probabilities) will be computed.
        return_proba : bool
            Also return the class probabilities for `data_set`.
        embedding_method : str, default='pooled'
            Embedding method to use ['pooled', 'gradient'].

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

        dataset_iter = dataloader(data_set.data, self.mini_batch_size, self._create_collate_fn(), train=False)

        tensors = []
        proba = []

        requires_grad = embedding_method == self.EMBEDDING_METHOD_GRADIENT
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
                        proba.extend(F.softmax(logits, dim=1).detach().to('cpu').tolist())
                    tensors.extend(embeddings)
        if return_proba:
            return np.array(tensors), np.array(proba)

        return np.array(tensors)

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

        batch_len = text.size(0)

        sm = F.softmax(logits, dim=1)
        sm_t = torch.t(sm)

        reduction_tmp = self.criterion.reduction
        self.criterion.reduction = 'none'

        modules = dict({name: module for name, module in self.model.named_modules()})
        grad_size = module_selector(modules).weight.flatten().size(0)

        arr = torch.empty(batch_len, grad_size * self.num_classes)
        for c in range(self.num_classes):
            loss = self.criterion(sm, torch.LongTensor([c] * batch_len).to(self.device))

            with torch.autocast(device_type=self.amp_args.device_type, dtype=self.amp_args.dtype,
                                enabled=False):
                for k in range(batch_len):
                    self.model.zero_grad()
                    loss[k].backward(retain_graph=True)

                    modules = dict({name: module for name, module in self.model.named_modules()})
                    params = module_selector(modules).weight.grad.flatten()

                    with torch.no_grad():
                        sm_prob = sm_t[c][k]
                        if c == best_label[k]:
                            arr[k, grad_size*c:grad_size*(c+1)] = (1-sm_prob)*params
                        else:
                            arr[k, grad_size*c:grad_size*(c+1)] = -1*sm_prob*params

        self.criterion.reduction = reduction_tmp

        return arr


class KimCNNClassifier(KimCNNEmbeddingMixin, PytorchClassifier):

    def __init__(self, num_classes, multi_label=False, embedding_matrix=None, device=None,
                 num_epochs=10, mini_batch_size=25, lr=0.001, max_seq_len=60, out_channels=100,
                 filter_padding=0, dropout=0.5, validation_set_size=0.1, padding_idx=0,
                 kernel_heights=[3, 4, 5], show_progress_bar=None, class_weight=None, amp_args=None,
                 compile_model=False, verbosity=VERBOSITY_MORE_VERBOSE):
        """
        num_classes : int
            Number of classes.
        multi_label : bool, default=False
            If `False`, the classes are mutually exclusive, i.e. the prediction step results in
            exactly one predicted label per instance.
        embedding_matrix : torch.FloatTensor
            A tensor of embeddings in the shape of (vocab_size, embedding_size).
        device : str or torch.device, default=None
            Torch device on which the computation will be performed.
        num_epochs : int, default=10
            Epochs to train.
        mini_batch_size : int, default=12
            Size of mini batches during training.
        lr : float, default=2e-5
            Learning rate.
        max_seq_len : int
            Maximum sequence length.
        out_channels : int
            Number of output channels.
        filter_padding : int
            Size of the padding to add before and after the sequence before applying the filters.
        dropout : float
            Dropout probability for the final layer in KimCNN.
        validation_set_size : float, default=0.1
            The size of the validation set as a fraction of the training set.
        padding_idx : int
            Index of the padding token (as given by the `vocab`).
        kernel_heights : list of int
            Kernel sizes.
        class_weight : 'balanced' or None, default=None
            If 'balanced', then the loss function is weighted inversely proportional to the
            label distribution to the current train set.
        show_progress_bar : bool or None
            Determines whether progress bars are shown. If none, the small-text default is used.
        amp_args : AMPArguments, default=None
            Configures the use of Automatic Mixed Precision (AMP).

            .. seealso:: :py:class:`~small_text.integrations.pytorch.classifiers.base.AMPArguments`
            .. versionadded:: 2.0.0

        compile_model : bool, default=False
            Compiles the model (using `torch.compile`) if `True` and PyTorch version is greater than or equal 2.0.0.

            .. versionadded:: 2.0.0
        """
        super().__init__(multi_label=multi_label, device=device, mini_batch_size=mini_batch_size,
                         amp_args=amp_args)

        with verbosity_logger():
            self.logger = logging.getLogger(__name__)
            self.logger.verbosity = verbosity

        if embedding_matrix is None:
            raise ValueError('This implementation requires an embedding matrix.')

        # Training parameters
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

        # KimCNN (pytorch model) parameters
        self.max_seq_len = max_seq_len
        self.out_channels = out_channels
        self.filter_padding = filter_padding
        self.dropout = dropout
        self.validation_set_size = validation_set_size
        self.embedding_matrix = embedding_matrix
        self.padding_idx = padding_idx
        self.kernel_heights = kernel_heights

        self.model = None
        self.model_selection = None
        self.enc_ = None

    def fit(self, train_set, validation_set=None, weights=None, early_stopping=None,
            model_selection=None, optimizer=None, scheduler=None):
        """Trains the model using the given train set.

        Parameters
        ----------
        train_set : PytorchTextClassificationDataset
            The dataset used for training the model.
        validation_set : PytorchTextClassificationDataset
            A validation set used for validation during training, or `None`. If `None`, the fit
            operation will split apart a subset of the train set as a validation set, whose size
            is set by `self.validation_set_size`.
        weights : np.ndarray[np.float32] or None, default=None
            Sample weights or None.
        early_stopping : EarlyStoppingHandler or 'none'
            A strategy for early stopping. Passing 'none' disables early stopping.
        model_selection : ModelSelectionHandler or None, default=None
            A model selection handler. Passing 'none' disables model selection.
        optimizer : torch.optim.optimizer.Optimizer or None, default=None
            A pytorch optimizer.
        scheduler :torch.optim._LRScheduler or None, default=None
            A pytorch scheduler.

        Returns
        -------
        self : KimCNNClassifier
            Returns the current classifier with a fitted model.
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
        self.model = KimCNN(vocab_size, self.max_seq_len, num_classes=self.num_classes,
                            dropout=self.dropout, out_channels=self.out_channels,
                            embedding_matrix=self.embedding_matrix,
                            embed_dim=embed_dim,
                            freeze_embedding_layer=False, padding_idx=self.padding_idx,
                            kernel_heights=self.kernel_heights)

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
        return partial(kimcnn_collate_fn, multi_label=self.multi_label,
                       num_classes=self.num_classes, use_sample_weights=use_sample_weights,
                       padding_idx=self.padding_idx, max_seq_len=self.max_seq_len,
                       filter_padding=self.filter_padding)

    def _train_func(self, sub_train_, weights, optimizer, scheduler, scaler, amp_args):

        train_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        train_acc = torch.tensor(0., dtype=torch.float32, device=self.device)

        data = sub_train_.data
        if weights is not None:
            data = [d + (weights[i],) for i, d in enumerate(data)]

        train_iter = dataloader(data, self.mini_batch_size,
                                self._create_collate_fn(use_sample_weights=weights is not None))

        for i, (text, cls, weight) in enumerate(train_iter):
            with torch.autocast(enabled=amp_args.use_amp, device_type=amp_args.device_type,
                                dtype=amp_args.dtype):
                loss, acc = self._train_single_batch(text, cls, weight, optimizer, scaler)
                scheduler.step()

                train_loss += loss.detach()
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
            if self.num_classes == 2:
                target = F.one_hot(cls, 2).float()
            else:
                target = cls
        loss = self.criterion(output, target)
        return loss

    def validate(self, validation_set):
        """Obtains validation scores (loss, accuracy) for the given validation set.

        Parameters
        ----------
        validation_set : PytorchTextClassificationDataset
            Validation set.

        Returns
        -------
        validation_loss : float
            Validation loss.
        validation_acc : float
            Validation accuracy.
        """
        with torch.no_grad():
            with torch.autocast(device_type=self.amp_args.device_type, dtype=self.amp_args.dtype,
                                enabled=self.amp_args.use_amp):

                valid_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
                acc = torch.tensor(0., dtype=torch.float32, device=self.device)

                self.model.eval()
                valid_iter = dataloader(validation_set.data, self.mini_batch_size, self._create_collate_fn(),
                                        train=False)

                for x, cls, weight, *_ in valid_iter:
                    x, cls, weight = x.to(self.device), cls.to(self.device), weight.to(self.device)

                    with torch.no_grad():
                        if self.num_classes == 2:
                            target = F.one_hot(cls, 2).float()
                        else:
                            target = cls

                        output = self.model(x)
                        loss = self.criterion(output, target)
                        loss = loss * weight
                        loss = loss.mean()

                        valid_loss += loss.detach()

                        acc += self.sum_up_accuracy_(output, cls)
                        del output, x, cls

                return valid_loss.item() / len(validation_set), acc.item() / len(validation_set)

    def predict(self, dataset, return_proba=False):
        """Predicts the labels for the given dataset.

        Parameters
        ----------
        dataset : PytorchTextClassificationDataset
            A dataset on whose instances predictions are made.
        return_proba : bool
            If True, additionally returns the confidence distribution over all classes.

        Returns
        -------
        predictions : np.ndarray[np.int32] or csr_matrix[np.int32]
            List of predictions if the classifier was fitted on single-label data,
            otherwise a sparse matrix of predictions.
        probas : np.ndarray[np.float32] (optional)
            List of probabilities (or confidence estimates) if `return_proba` is True.
        """
        return super().predict(dataset, return_proba=return_proba)

    def predict_proba(self, dataset, dropout_sampling=1):
        """Predicts the label distributions.

        Parameters
        ----------
        dataset : PytorchTextClassificationDataset
            A dataset whose labels will be predicted.
        dropout_sampling : int
            If `dropout_sampling > 1` then all dropout modules will be enabled during prediction and
            multiple rounds of predictions will be sampled for each instance.

        Returns
        -------
        scores : np.ndarray
            Confidence score distribution over all classes of shape (num_samples, num_classes).
            If `dropout_sampling > 1` then the shape is (num_samples, dropour_samples, num_classes).
        """
        return super().predict_proba(dataset, dropout_sampling=dropout_sampling)

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
            for text,  *_ in dataset_iter:
                batch_size, vector_len = text.shape
                full_size = batch_size * dropout_samples
                text = text.to(self.device)
                text = text.repeat(1, dropout_samples).resize(full_size, vector_len)

                output = self.model.forward(text)

                prediction_for_batch = logits_transform(output)
                prediction_for_batch = prediction_for_batch.unsqueeze(dim=1)\
                    .resize(batch_size, dropout_samples, self.num_classes)

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
