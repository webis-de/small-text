import datetime
import logging
import tempfile
import warnings

import numpy as np

from functools import partial
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer

from small_text.classifiers.classification import EmbeddingMixin
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.classification import empty_result, get_splits
from small_text.utils.context import build_pbar_context
from small_text.utils.data import check_training_data
from small_text.utils.data import list_length
from small_text.utils.datetime import format_timedelta
from small_text.utils.labels import csr_to_list, get_num_labels
from small_text.utils.logging import verbosity_logger, VERBOSITY_MORE_VERBOSE
from small_text.utils.system import get_tmp_dir_base

try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    from torch.optim import AdamW
    from transformers import logging as transformers_logging
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    from small_text.integrations.pytorch.classifiers.base import (
        check_optimizer_and_scheduler_config,
        PytorchClassifier
    )
    from small_text.integrations.pytorch.model_selection import Metric, PytorchModelSelection
    from small_text.integrations.pytorch.utils.data import dataloader
    from small_text.integrations.transformers.datasets import TransformersDataset
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


def transformers_collate_fn(batch, enc=None):
    with torch.no_grad():
        text = torch.cat([entry[TransformersDataset.INDEX_TEXT] for entry in batch], dim=0)
        masks = torch.cat([entry[TransformersDataset.INDEX_MASK] for entry in batch], dim=0)
        if enc is not None:
            labels = [entry[TransformersDataset.INDEX_LABEL] for entry in batch]
            multi_hot = enc.transform(labels)
            label = torch.tensor(multi_hot, dtype=float)
        else:
            label = torch.tensor([entry[TransformersDataset.INDEX_LABEL] for entry in batch])

    return text, masks, label


class FineTuningArguments(object):
    """
    Arguments to enable and configure gradual unfreezing and discriminative learning rates as used in
    Universal Language Model Fine-tuning (ULMFiT) [HR18]_.
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
        self.cut_fraction = cut_fraction


class TransformerModelArguments(object):

    def __init__(self, model, tokenizer=None, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        if self.tokenizer is None:
            self.tokenizer = model
        if self.config is None:
            self.config = model


def _get_layer_params(model, base_lr, fine_tuning_arguments):

    layerwise_gradient_decay = fine_tuning_arguments.layerwise_gradient_decay

    params = []

    base_model = getattr(model, model.base_model_prefix)
    if hasattr(base_model, 'encoder'):
        layers = base_model.encoder.layer
    else:
        layers = base_model.transformer.layer

    total_layers = len(layers)

    use_gradual_unfreezing = isinstance(fine_tuning_arguments.gradual_unfreezing, int) and \
        fine_tuning_arguments.gradual_unfreezing > 0

    start_layer = 0 if not use_gradual_unfreezing else total_layers-fine_tuning_arguments.gradual_unfreezing
    num_layers = total_layers - start_layer

    for i in range(start_layer, total_layers):
        lr = base_lr if not layerwise_gradient_decay else base_lr * layerwise_gradient_decay ** (
                    num_layers - i)
        params.append({
            'params': layers[i].parameters(),
            'lr': lr
        })

    return params


class TransformerBasedEmbeddingMixin(EmbeddingMixin):

    EMBEDDING_METHOD_AVG = 'avg'
    EMBEDDING_METHOD_CLS_TOKEN = 'cls'

    def embed(self, data_set, return_proba=False, embedding_method=EMBEDDING_METHOD_AVG,
              hidden_layer_index=-1, pbar='tqdm'):
        """
        Embeds each sample in the given `data_set`.

        The embedding is created by using hidden representation from the transformer model's
        representation in the hidden layer at the given `hidden_layer_index`.

        Parameters
        ----------
        return_proba : bool
            Also return the class probabilities for `data_set`.
        embedding_method : str
            Embedding method to use [avg, cls].
        hidden_layer_index : int
            Index of the hidden layer.
        pbar : 'tqdm' or None
            Displays a progress bar if 'tqdm' is passed.

        Returns
        -------
        embeddings : np.ndarray
            Embeddings in the shape (N, hidden_layer_dimensionality).
        proba : np.ndarray
            Class probabilities for `data_set` (only if `return_predictions` is `True`).
        """

        if self.model is None:
            raise ValueError('Model is not trained. Please call fit() first.')

        self.model.eval()

        train_iter = dataloader(data_set.data, self.mini_batch_size, self._create_collate_fn(),
                                train=False)

        tensors = []
        predictions = []

        with build_pbar_context(pbar, tqdm_kwargs={'total': list_length(data_set)}) as pbar:
            for batch in train_iter:
                batch_len, logits = self._create_embeddings(tensors,
                                                            batch,
                                                            embedding_method=embedding_method,
                                                            hidden_layer_index=hidden_layer_index)
                pbar.update(batch_len)
                if return_proba:
                    predictions.extend(F.softmax(logits, dim=1).detach().to('cpu').tolist())

        if return_proba:
            return np.array(tensors), np.array(predictions)

        return np.array(tensors)

    def _create_embeddings(self, tensors, batch, embedding_method='avg', hidden_layer_index=-1):

        text, masks, _ = batch
        text = text.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)

        outputs = self.model(text,
                             token_type_ids=None,
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

        tensors.extend(representation.detach().to('cpu', non_blocking=True).numpy())

        return text.size(0), outputs.logits


class TransformerBasedClassification(TransformerBasedEmbeddingMixin, PytorchClassifier):

    def __init__(self, transformer_model, num_classes, multi_label=False, num_epochs=10, lr=2e-5,
                 mini_batch_size=12, validation_set_size=0.1, validations_per_epoch=1,
                 no_validation_set_action='sample', early_stopping_no_improvement=5,
                 early_stopping_acc=-1, model_selection=True, fine_tuning_arguments=None,
                 device=None, memory_fix=1, class_weight=None, verbosity=VERBOSITY_MORE_VERBOSE,
                 cache_dir='.active_learning_lib_cache/'):
        """
        Parameters
        ----------
        transformer_model : TransformerModelArguments
            Settings for transformer model, tokenizer and config.
        num_classes : int
            Number of classes.
        num_epochs : int
            Epochs to train.
        lr : float
            Learning rate.
        mini_batch_size : int
            Size of mini batches during training.

        validation_set_size : float
            The sizes of the validation as a fraction of the training set if no validation set
            is passed and `no_validation_set_action` is set to 'sample'.
        validations_per_epoch : int
            Defines how of the validation set is evaluated during the training of a single epoch.
        no_validation_set_action : {'sample', 'none}
            Defines what should be done of no validation set is given.
        early_stopping_no_improvement :

        early_stopping_acc :

        model_selection :

        fine_tuning_arguments : FineTuningArguments

        device : str or torch.device
            Torch device on which the computation will be performed.
        memory_fix : int
            If this value if greater zero, every `memory_fix`-many epochs the cuda cache will be
            emptied to force unused GPU memory being released.

        class_weight : 'balanced' or None

        """
        super().__init__(multi_label=multi_label, device=device, mini_batch_size=mini_batch_size)

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
        # 'sample' or 'none'
        self.no_validation_set_action = no_validation_set_action

        # Huggingface
        self.transformer_model = transformer_model

        # Other
        self.early_stopping_no_improvement = early_stopping_no_improvement
        self.early_stopping_acc = early_stopping_acc
        self.class_weight = class_weight

        self.model_selection = model_selection
        self.fine_tuning_arguments = fine_tuning_arguments

        self.memory_fix = memory_fix
        self.verbosity = verbosity
        self.cache_dir = cache_dir

        self.model = None
        self.model_selection_manager = None

        self.enc_ = None

    def fit(self, train_set, validation_set=None, optimizer=None, scheduler=None):
        """
        Trains the model using the given train set.

        Parameters
        ----------
        train_set : TransformersDataset
            Training set.
        validation_set : TransformersDataset
            A validation set used for validation during training, or `None`. If `None`, the fit
            operation will split apart a subset of the trainset as a validation set, whose size
            is set by `self.validation_set_size`.
        optimizer : torch.optim.optimizer.Optimizer
            A pytorch optimizer.
        scheduler :torch.optim._LRScheduler
            A pytorch scheduler.

        Returns
        -------
        self : TransformerBasedClassification
            Returns the current classifier with a fitted model.
        """
        check_training_data(train_set, validation_set)
        optimizer_or_scheduler_given = optimizer is not None or scheduler is not None
        if self.fine_tuning_arguments is not None and optimizer_or_scheduler_given:
            raise ValueError('When fine_tuning_arguments are provided you cannot pass '
                             'optimizer and scheduler to fit()')

        sub_train, sub_valid = get_splits(train_set, validation_set, multi_label=self.multi_label,
                                          validation_set_size=self.validation_set_size)

        fit_optimizer = optimizer if optimizer is not None else self.optimizer
        fit_scheduler = scheduler if scheduler is not None else self.scheduler

        if self.multi_label:
            self.enc_ = MultiLabelBinarizer()
            labels = csr_to_list(sub_train.y)
            self.enc_ = self.enc_.fit(labels)

        self.class_weights_ = self.initialize_class_weights(sub_train)
        self.criterion = self.get_default_criterion()

        return self._fit_main(sub_train, sub_valid, fit_optimizer, fit_scheduler)

    def _fit_main(self, sub_train, sub_valid, optimizer, scheduler):
        if self.model is None:
            encountered_num_classes = get_num_labels(sub_train.y)

            if self.num_classes is None:
                self.num_classes = encountered_num_classes

            if self.num_classes != encountered_num_classes:
                raise ValueError('Conflicting information about the number of classes: '
                                 'expected: {}, encountered: {}'.format(self.num_classes,
                                                                        encountered_num_classes))

            self.initialize_transformer(self.cache_dir)

        check_optimizer_and_scheduler_config(optimizer, scheduler)
        scheduler = scheduler if scheduler is None else 'linear'

        optimizer, scheduler = self._get_optimizer_and_scheduler(optimizer,
                                                                 scheduler,
                                                                 self.num_epochs,
                                                                 sub_train)
        self.model = self.model.to(self.device)

        with tempfile.TemporaryDirectory(dir=get_tmp_dir_base()) as tmp_dir:
            res = self._train(sub_train, sub_valid, tmp_dir, optimizer, scheduler)
            self._perform_model_selection(sub_valid)

        return res

    def initialize_transformer(self, cache_dir):

        self.config = AutoConfig.from_pretrained(
            self.transformer_model.config,
            num_labels=self.num_classes,
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.transformer_model.tokenizer,
            cache_dir=cache_dir,
        )

        # Suppress "Some weights of the model checkpoint at [model name] were not [...]"-warnings
        previous_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.transformer_model.model,
            from_tf=False,
            config=self.config,
            cache_dir=cache_dir,
        )
        transformers_logging.set_verbosity(previous_verbosity)

    def _default_optimizer(self, base_lr):

        if self.fine_tuning_arguments is not None:
            params = _get_layer_params(self.model, self.lr, self.fine_tuning_arguments)
        else:
            params = [param for param in self.model.parameters() if param.requires_grad]

        return params, AdamW(params, lr=base_lr, eps=1e-8)

    def _train(self, sub_train, sub_valid, tmp_dir, optimizer, scheduler):

        metrics = [Metric('valid_loss', True), Metric('valid_acc', False),
                   Metric('train_loss', True), Metric('train_acc', False)]
        self.model_selection_manager = PytorchModelSelection(Path(tmp_dir), metrics)

        start_epoch = 0
        self._train_loop(sub_train, sub_valid, optimizer, scheduler, start_epoch, self.num_epochs,
                         self.model_selection_manager)

        return self

    def _train_loop(self, sub_train, sub_valid, optimizer, scheduler, start_epoch, num_epochs,
                    model_selection_manager):

        min_loss = float('inf')
        no_loss_reduction = 0

        stopped = False

        for epoch in range(start_epoch, num_epochs):
            start_time = datetime.datetime.now()

            if self.memory_fix and (epoch + 1) % self.memory_fix == 0:
                torch.cuda.empty_cache()

            self.model.train()

            # TODO: extract this block after introducing a shared return type
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

                train_loss, train_acc, valid_loss, valid_acc = self._train_loop_process_batches(
                    sub_train,
                    optimizer,
                    scheduler,
                    validate_every=validate_every,
                    validation_set=sub_valid)
            else:
                train_loss, train_acc = self._train_loop_process_batches(
                    sub_train,
                    optimizer,
                    scheduler)

                if sub_valid is not None:
                    valid_loss, valid_acc = self.validate(sub_valid)

            timedelta = datetime.datetime.now() - start_time

            self._log_epoch(epoch, timedelta, sub_train, sub_valid, train_acc, train_loss,
                            valid_acc, valid_loss)

            if sub_valid is not None:
                if self.early_stopping_no_improvement > 0:
                    if valid_loss < min_loss:
                        no_loss_reduction = 0
                        min_loss = valid_loss
                    else:
                        no_loss_reduction += 1

                        if no_loss_reduction >= self.early_stopping_no_improvement:
                            logging.info(f'Early stopping after {epoch + 1} epochs')
                            stopped = True

                if not stopped and self.early_stopping_acc > 0:
                    if train_acc > self.early_stopping_acc:
                        logging.info(f'Early stopping due to high train acc: {train_acc}')
                        stopped = True

                model_selection_manager.add_model(self.model, epoch + 1, valid_acc=valid_acc,
                                                  valid_loss=valid_loss, train_acc=train_acc,
                                                  train_loss=train_loss)

            if stopped:
                break

    def _train_loop_process_batches(self, sub_train_, optimizer, scheduler, validate_every=None,
                                    validation_set=None):

        train_loss = 0.
        train_acc = 0.
        valid_losses = []
        valid_accs = []

        train_iter = dataloader(sub_train_.data, self.mini_batch_size, self._create_collate_fn())

        for i, (x, masks, cls) in enumerate(train_iter):
            loss, acc = self._train_single_batch(x, masks, cls, optimizer)
            scheduler.step()

            train_loss += loss
            train_acc += acc

            if validate_every and i % validate_every == 0:
                valid_loss, valid_acc = self.validate(validation_set)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)

        if validate_every:
            return train_loss / len(sub_train_), train_acc / len(sub_train_), \
                   np.mean(valid_losses), np.mean(valid_accs)
        else:
            return train_loss / len(sub_train_), train_acc / len(sub_train_)

    def _create_collate_fn(self):
        return partial(transformers_collate_fn, enc=self.enc_)

    def _train_single_batch(self, x, masks, cls, optimizer):

        train_loss = 0.
        train_acc = 0.

        optimizer.zero_grad()

        x, masks, cls = x.to(self.device), masks.to(self.device), cls.to(self.device)
        outputs = self.model(x, attention_mask=masks)

        logits, loss = self._compute_loss(cls, outputs)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.detach().item()
        train_acc += self.sum_up_accuracy_(logits, cls)

        del x, masks, cls, loss, outputs

        return train_loss, train_acc

    def _compute_loss(self, cls, outputs):
        if self.num_classes == 2:
            logits = outputs.logits
            target = F.one_hot(cls, 2).float()
        else:
            logits = outputs.logits.view(-1, self.num_classes)
            target = cls
        loss = self.criterion(logits, target)

        return logits, loss

    def _perform_model_selection(self, sub_valid):
        if sub_valid is not None:
            if self.model_selection:
                self._select_best_model()
            else:
                self._select_last_model()

    def _select_best_model(self):
        model_path, _ = self.model_selection_manager.select_best()
        self.model.load_state_dict(torch.load(model_path))

    def _select_last_model(self):
        model_path, _ = self.model_selection_manager.select_last()
        self.model.load_state_dict(torch.load(model_path))

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

        valid_loss = 0.
        acc = 0.

        self.model.eval()
        valid_iter = dataloader(validation_set.data, self.mini_batch_size, self._create_collate_fn(),
                                train=False)

        for x, masks, cls in valid_iter:
            x, masks, cls = x.to(self.device), masks.to(self.device), cls.to(self.device)

            with torch.no_grad():
                outputs = self.model(x, attention_mask=masks)
                _, loss = self._compute_loss(cls, outputs)

                valid_loss += loss.item()
                acc += self.sum_up_accuracy_(outputs.logits, cls)
                del outputs, x, masks, cls

        return valid_loss / len(validation_set), acc / len(validation_set)

    def predict(self, data_set, return_proba=False):
        """
        Parameters
        ----------
        data_set : small_text.integrations.transformers.TransformerDataset
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
        return super().predict(data_set, return_proba=return_proba)

    def predict_proba(self, test_set):
        if len(test_set) == 0:
            return empty_result(self.multi_label, self.num_classes, return_prediction=False,
                                return_proba=True)

        self.model.eval()
        test_iter = dataloader(test_set.data, self.mini_batch_size, self._create_collate_fn(),
                               train=False)

        predictions = []
        logits_transform = torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)

        with torch.no_grad():
            for text, masks, _ in test_iter:
                text, masks = text.to(self.device), masks.to(self.device)
                outputs = self.model(text, attention_mask=masks)

                predictions += logits_transform(outputs.logits).to('cpu').tolist()
                del text, masks

        return np.array(predictions)

    def __del__(self):
        try:
            attrs = ['criterion', 'optimizer', 'scheduler', 'model', 'tokenizer', 'config']
            for attr in attrs:
                delattr(self, attr)
        except Exception:
            pass
