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
from small_text.utils.classification import empty_result, get_splits
from small_text.utils.context import build_pbar_context
from small_text.utils.data import check_training_data, list_length
from small_text.utils.labels import csr_to_list, get_num_labels
from small_text.utils.datetime import format_timedelta
from small_text.utils.logging import verbosity_logger, VERBOSITY_MORE_VERBOSE


try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from torch.optim import Adadelta

    from small_text.integrations.pytorch.classifiers.base import (
        check_optimizer_and_scheduler_config
    )
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from small_text.integrations.pytorch.model_selection import Metric, PytorchModelSelection
    from small_text.integrations.pytorch.utils.data import dataloader
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


def kimcnn_collate_fn(batch, enc=None, max_seq_len=60, padding_idx=0, filter_padding=0):

    def prepare_tensor(t):
        t_sub = t[:max_seq_len-2*filter_padding]
        return torch.cat([t_sub.new_zeros(filter_padding) + padding_idx,
                          t_sub,
                          t_sub.new_zeros(max_seq_len - 2*filter_padding - t_sub.size(0)) + padding_idx,
                          t_sub.new_zeros(filter_padding) + padding_idx],
                         0)

    if enc is not None:
        labels = [entry[PytorchTextClassificationDataset.INDEX_LABEL] for entry in batch]
        multi_hot = enc.transform(labels)
        label = torch.tensor(multi_hot, dtype=float)
    else:
        label = torch.tensor([entry[PytorchTextClassificationDataset.INDEX_LABEL] for entry in batch])
    text = torch.stack([prepare_tensor(t) for t, _ in batch], 0)

    return text, label


class KimCNNEmbeddingMixin(EmbeddingMixin):

    EMBEDDING_METHOD_POOLED = 'pooled'
    EMBEDDING_METHOD_GRADIENT = 'gradient'

    def embed(self, data_set, return_proba=False, embedding_method=EMBEDDING_METHOD_POOLED,
              module_selector=lambda x: x['fc'], pbar='tqdm'):

        if self.model is None:
            raise ValueError('Model is not trained. Please call fit() first.')

        self.model.eval()

        dataset_iter = dataloader(data_set.data, self.mini_batch_size, self._create_collate_fn(),
                                  train=False)

        tensors = []
        proba = []
        with build_pbar_context(pbar, tqdm_kwargs={'total': list_length(data_set)}) as pbar:
            for text, _ in dataset_iter:
                batch_len = text.size(0)
                text = text.to(self.device, non_blocking=True)

                if embedding_method == self.EMBEDDING_METHOD_POOLED:
                    embedded = self.model._forward_pooled(text)
                    tensors.extend(embedded.detach().to('cpu', non_blocking=True).numpy())

                    if return_proba:
                        sm = F.softmax(self.model._dropout_and_fc(embedded), dim=1)
                        proba.extend(sm.detach().to('cpu').tolist())
                    pbar.update(batch_len)

                elif embedding_method == self.EMBEDDING_METHOD_GRADIENT:
                    best_label, sm = self.get_best_and_softmax(proba, text)
                    self.create_embedding(best_label, sm, module_selector, tensors, text)
                    pbar.update(batch_len)
                else:
                    raise ValueError(f'Invalid embedding method: {embedding_method}')

        if return_proba:
            return np.array(tensors), np.array(proba)

        return np.array(tensors)

    def get_best_and_softmax(self, proba, text):

        self.model.zero_grad()

        output = self.model(text)

        sm = F.softmax(output, dim=1)
        with torch.no_grad():
            best_label = torch.argmax(sm, dim=1)
        proba.extend(sm.detach().to('cpu', non_blocking=True).numpy())

        return best_label, sm

    def create_embedding(self, best_label, sm, module_selector, tensors, text):

        batch_len = text.size(0)
        sm_t = torch.t(sm)

        reduction_tmp = self.criterion.reduction
        self.criterion.reduction = 'none'

        modules = dict({name: module for name, module in self.model.named_modules()})
        grad = module_selector(modules).weight.grad
        grad_size = grad.flatten().size(0)

        arr = torch.empty(batch_len, grad_size * self.num_classes)
        for c in range(self.num_classes):
            loss = self.criterion(sm, torch.LongTensor([c] * batch_len).to(self.device))

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

        tensors.extend(arr.detach().to('cpu', non_blocking=True).numpy())
        self.criterion.reduction = reduction_tmp

        return batch_len


class KimCNNClassifier(KimCNNEmbeddingMixin, PytorchClassifier):

    def __init__(self, num_classes, multi_label=False, embedding_matrix=None, device=None,
                 num_epochs=10, mini_batch_size=25, lr=0.001, max_seq_len=60, out_channels=100,
                 filter_padding=0, dropout=0.5, validation_set_size=0.1, padding_idx=0,
                 kernel_heights=[3, 4, 5], early_stopping=5, early_stopping_acc=0.98,
                 class_weight=None, verbosity=VERBOSITY_MORE_VERBOSE):

        super().__init__(multi_label=multi_label, device=device, mini_batch_size=mini_batch_size)

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

        # KimCNN (pytorch model) parameters
        self.max_seq_len = max_seq_len
        self.out_channels = out_channels
        self.filter_padding = filter_padding
        self.dropout = dropout
        self.validation_set_size = validation_set_size
        self.embedding_matrix = embedding_matrix
        self.padding_idx = padding_idx
        self.kernel_heights = kernel_heights

        self.early_stopping = early_stopping
        self.early_stopping_acc = early_stopping_acc

        self.model = None
        self.model_selection = None
        self.enc_ = None

    def fit(self, train_set, validation_set=None, optimizer=None, scheduler=None):
        """
        Trains the model using the given train set.

        Parameters
        ----------
        train_set : PytorchTextClassificationDataset
            The dataset used for training the model.
        validation_set : PytorchTextClassificationDataset
            A validation set used for validation during training, or `None`. If `None`, the fit
            operation will split apart a subset of the trainset as a validation set, whose size
            is set by `self.validation_set_size`.
        optimizer : torch.optim.optimizer.Optimizer
            A pytorch optimizer.
        scheduler :torch.optim._LRScheduler
            A pytorch scheduler.

        Returns
        -------
        self : KimCNNClassifier
            Returns the current classifier with a fitted model.
        """
        check_training_data(train_set, validation_set)

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

            self.initialize_kimcnn_model(sub_train)

        check_optimizer_and_scheduler_config(optimizer, scheduler)
        scheduler = scheduler if scheduler is None else None

        optimizer, scheduler = self._get_optimizer_and_scheduler(optimizer,
                                                                 scheduler,
                                                                 self.num_epochs,
                                                                 sub_train)

        self.model = self.model.to(self.device)
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self._train(sub_train, sub_valid, tmp_dir, optimizer, scheduler)

            model_path, _ = self.model_selection.select_best()
            self.model.load_state_dict(torch.load(model_path))

        return res

    def initialize_kimcnn_model(self, sub_train):
        vocab_size = len(sub_train.vocab)

        embed_dim = self.embedding_matrix.shape[1]
        self.model = KimCNN(vocab_size, self.max_seq_len, num_classes=self.num_classes,
                            dropout=self.dropout, out_channels=self.out_channels,
                            embedding_matrix=self.embedding_matrix,
                            embed_dim=embed_dim,
                            freeze_embedding_layer=False, padding_idx=self.padding_idx,
                            kernel_heights=self.kernel_heights)

    def _default_optimizer(self, base_lr):
        params = [param for param in self.model.parameters() if param.requires_grad]
        return params, Adadelta(params, lr=base_lr, eps=1e-8)

    def _train(self, sub_train, sub_valid, tmp_dir, optimizer, scheduler):

        min_loss = float('inf')
        no_loss_reduction = 0

        metrics = [Metric('valid_loss', True), Metric('valid_acc', False),
                   Metric('train_loss', True), Metric('train_acc', False)]
        self.model_selection = PytorchModelSelection(tmp_dir, metrics=metrics)

        for epoch in range(self.num_epochs):
            start_time = datetime.datetime.now()

            self.model.train()
            train_loss, train_acc = self._train_func(sub_train, optimizer, scheduler)

            self.model.eval()
            valid_loss, valid_acc = self.validate(sub_valid)
            self.model_selection.add_model(self.model, epoch+1, valid_acc=valid_acc,
                                           valid_loss=valid_loss, train_acc=train_acc,
                                           train_loss=train_loss)

            timedelta = datetime.datetime.now() - start_time

            self.logger.info(f'Epoch: {epoch+1} | {format_timedelta(timedelta)}\n'
                             f'\tTrain Set Size: {len(sub_train)}\n'
                             f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}% (train)\n'
                             f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}% (valid)',
                             verbosity=VERBOSITY_MORE_VERBOSE)

            if self.early_stopping > 0:
                if valid_loss < min_loss:
                    no_loss_reduction = 0
                    min_loss = valid_loss
                else:
                    no_loss_reduction += 1

                    if no_loss_reduction >= self.early_stopping:
                        print('\nEarly stopping after %s epochs' % (epoch+1))
                        return self

            if self.early_stopping_acc > 0:
                if train_acc > self.early_stopping_acc:
                    print('\nEarly stopping due to high train acc: %s' % (train_acc))
                    return self

        return self

    def _create_collate_fn(self):
        return partial(kimcnn_collate_fn, enc=self.enc_, padding_idx=self.padding_idx,
                       max_seq_len=self.max_seq_len, filter_padding=self.filter_padding)

    def _train_func(self, sub_train_, optimizer, scheduler):

        train_loss = 0.
        train_acc = 0.

        train_iter = dataloader(sub_train_.data, self.mini_batch_size, self._create_collate_fn())

        for i, (text, cls) in enumerate(train_iter):
            loss, acc = self._train_single_batch(text, cls, optimizer)
            scheduler.step()

            train_loss += loss
            train_acc += acc

        return train_loss / len(sub_train_), train_acc / len(sub_train_)

    def _train_single_batch(self, text, cls, optimizer):

        train_loss = 0.
        train_acc = 0.

        optimizer.zero_grad()

        text, cls = text.to(self.device), cls.to(self.device)
        output = self.model(text)

        with torch.no_grad():
            if self.num_classes == 2:
                target = F.one_hot(cls, 2).float()
            else:
                target = cls
        loss = self.criterion(output, target)

        loss.backward()

        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
            self.model.fc.weight.div_(torch.norm(self.model.fc.weight, dim=1, keepdim=True))

        optimizer.step()

        train_loss += loss.item()
        train_acc += self.sum_up_accuracy_(output, cls)

        del text, cls, output

        return train_loss, train_acc

    def validate(self, validation_set):
        """
        Obtains validation scores (loss, accuracy) for the given validation set.

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
        valid_loss = 0.
        acc = 0.

        self.model.eval()
        valid_iter = dataloader(validation_set.data, self.mini_batch_size, self._create_collate_fn(),
                                train=False)

        for x, cls in valid_iter:
            x, cls = x.to(self.device), cls.to(self.device)

            with torch.no_grad():
                if self.num_classes == 2:
                    target = F.one_hot(cls, 2).float()
                else:
                    target = cls

                output = self.model(x)
                loss = self.criterion(output, target)
                valid_loss += loss.item()

                acc += self.sum_up_accuracy_(output, cls)
                del output, x, cls

        return valid_loss / len(validation_set), acc / len(validation_set)

    def predict(self, data_set, return_proba=False):
        """
        Predicts the labels for the given dataset.

        Parameters
        ----------
        data_set : PytorchTextClassificationDataset
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
            for text, _ in test_iter:
                text = text.to(self.device)

                predictions += logits_transform(self.model.forward(text)).to('cpu').tolist()
                del text

        return np.array(predictions)

    def __del__(self):
        try:
            attrs = ['criterion', 'optimizer', 'scheduler', 'model']
            for attr in attrs:
                delattr(self, attr)
        except Exception:
            pass
