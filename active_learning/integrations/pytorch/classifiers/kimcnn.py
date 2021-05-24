import datetime
import logging
import tempfile

import numpy as np

from functools import partial

from active_learning.classifiers.classification import EmbeddingMixin
from active_learning.data.datasets import split_data
from active_learning.integrations.pytorch.classifiers.base import PytorchClassifier
from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError
from active_learning.integrations.pytorch.models.kimcnn import KimCNN
from active_learning.utils.context import build_pbar_context
from active_learning.utils.data import list_length
from active_learning.utils.datetime import format_timedelta


logger = logging.getLogger(__name__)


try:
    import torch
    import torch.nn.functional as F

    from torch import randperm

    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from active_learning.integrations.pytorch.model_selection import Metric, PytorchModelSelection
    from active_learning.integrations.pytorch.utils.data import dataloader
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


# TODO: pass filter_padding
def kimcnn_collate_fn(batch, max_seq_len=60, padding_idx=0, filter_padding=0):

    def prepare_tensor(t):
        t_sub = t[:max_seq_len-2*filter_padding]
        return torch.cat([t_sub.new_zeros(filter_padding) + padding_idx,
                          t_sub,
                          t_sub.new_zeros(max_seq_len - 2*filter_padding - t_sub.size(0)) + padding_idx,
                          t_sub.new_zeros(filter_padding) + padding_idx],
                         0)

    label = torch.tensor([entry[PytorchTextClassificationDataset.INDEX_LABEL] for entry in batch])
    text = torch.stack([prepare_tensor(t) for t, _ in batch], 0)

    return text, label


class KimCNNEmbeddingMixin(EmbeddingMixin):

    def embed(self, data_set, module_selector=lambda x: x['fc'], pbar='tqdm', **kwargs):

        if self.model is None:
            raise ValueError('Model is not trained. Please call fit() first.')

        dataset_iter = dataloader(data_set, self.mini_batch_size, self._create_collate_fn(),
                                  train=False)

        tensors = []
        with build_pbar_context(pbar, tqdm_kwargs={'total': list_length(data_set)}) as pbar:
            for text, _ in dataset_iter:
                batch_len = self.create_embedding(module_selector, tensors, text)
                pbar.update(batch_len)

        return np.array(tensors)

    def create_embedding(self, module_selector, tensors, text):

        text = text.to(self.device, non_blocking=True)

        self.model.zero_grad()
        batch_len = text.size(0)

        output = self.model(text)

        sm = F.softmax(output, dim=1)
        with torch.no_grad():
            best_label = torch.argmax(sm, dim=1)

        sm_t = torch.t(sm)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        modules = dict({name: module for name, module in self.model.named_modules()})
        grad = module_selector(modules).weight.grad
        grad_size = grad.flatten().size(0)

        arr = torch.empty(batch_len, grad_size * self.num_class)
        for c in range(self.num_class):
            loss = criterion(sm, torch.LongTensor([c] * batch_len).to(self.device))

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

        return batch_len


class KimCNNClassifier(KimCNNEmbeddingMixin, PytorchClassifier):

    def __init__(self, embedding_matrix=None, device=None, num_epochs=10, mini_batch_size=25, criterion=None, optimizer=None,
                 lr=0.001, max_seq_len=60, out_channels=100, dropout=0.5, validation_set_size=0.1,
                 padding_idx=0, kernel_heights=[3, 4, 5], early_stopping=5,
                 early_stopping_acc=0.98):

        super().__init__(device=device)

        if embedding_matrix is None:
            raise ValueError('This implementation requires an embedding matrix.')

        # Training parameters
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr

        # KimCNN (pytorch model) parameters
        self.max_seq_len = max_seq_len
        self.out_channels = out_channels
        self.dropout = dropout
        self.validation_set_size = validation_set_size
        self.embedding_matrix = embedding_matrix
        self.padding_idx = padding_idx
        self.kernel_heights = kernel_heights

        self.early_stopping = early_stopping
        self.early_stopping_acc = early_stopping_acc

        self.model = None
        self.model_selection = None

    def fit(self, train_set, validation_set=None, **kwargs):
        """
        Parameters
        ----------
        train_set : active_learning.integrations.pytorch.PytorchTextClassificationDataset
            Training set.

        Returns
        -------
        self : KimCNNClassifier
            Returns the current KimCNNClassification instance with a trained model.
        """
        # TODO: train_set.to('cpu')
        if (train_set.y == PytorchTextClassificationDataset.NO_LABEL).any():
            raise ValueError('Training labels must be greater or equal zero')
        if validation_set is not None and \
                (validation_set.y == PytorchTextClassificationDataset.NO_LABEL).any():
            raise ValueError('Validation set labels must be greater or equal zero')

        if validation_set is None:
            y = train_set.y.tolist()
            sub_train, sub_valid = split_data(train_set, y=y, strategy='stratified',
                                              validation_set_size=self.validation_set_size)
        else:
            sub_train, sub_valid = train_set, validation_set

        return self._fit_main(sub_train, sub_valid)

    def _fit_main(self, sub_train, sub_valid):
        embed_dim = self.embedding_matrix.shape[1]

        if self.model is None:
            vocab_size = len(sub_train.vocab)
            self.num_class = sub_train.target_labels.shape[0]
            self.model = KimCNN(vocab_size, self.max_seq_len, num_classes=self.num_class,
                                dropout=self.dropout, out_channels=self.out_channels,
                                embedding_matrix=self.embedding_matrix,
                                embed_dim=embed_dim,
                                freeze_embedding_layer=False, padding_idx=self.padding_idx,
                                kernel_heights=self.kernel_heights)

            if self.criterion is None:
                self.criterion = torch.nn.CrossEntropyLoss()
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model = self.model.to(self.device)
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self._train(sub_train, sub_valid, tmp_dir)

            model_path, _ = self.model_selection.select_best()
            self.model.load_state_dict(torch.load(model_path))

        return res

    def _train(self, sub_train, sub_valid, tmp_dir):

        min_loss = float('inf')
        no_loss_reduction = 0

        metrics = [Metric('valid_loss', True), Metric('valid_acc', False),
                   Metric('train_loss', True), Metric('train_acc', False)]
        self.model_selection = PytorchModelSelection(tmp_dir, metrics=metrics)

        for epoch in range(self.num_epochs):
            start_time = datetime.datetime.now()

            self.model.train()
            train_loss, train_acc = self._train_func(sub_train)

            self.model.eval()
            valid_loss, valid_acc = self.validate(sub_valid)
            self.model_selection.add_model(self.model, epoch+1, valid_acc=valid_acc,
                                           valid_loss=valid_loss, train_acc=train_acc,
                                           train_loss=train_loss)

            timedelta = datetime.datetime.now() - start_time

            logger.info(f'Epoch: {epoch+1} | {format_timedelta(timedelta)}')
            logger.info(f'\tTrain Set Size: {len(sub_train)}')
            logger.info(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}% (train)')
            logger.info(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}% (valid)')

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
        return partial(kimcnn_collate_fn, padding_idx=self.padding_idx,
                       max_seq_len=self.max_seq_len)

    def _train_func(self, sub_train_):

        train_loss = 0.
        train_acc = 0.

        train_iter = dataloader(sub_train_, self.mini_batch_size, self._create_collate_fn())

        for i, (text, cls) in enumerate(train_iter):
            loss, acc = self._train_single_batch(text, cls, self.optimizer, self.criterion)
            train_loss += loss
            train_acc += acc

        return train_loss / len(sub_train_), train_acc / len(sub_train_)

    def _train_single_batch(self, text, cls, optimizer, criterion):

        train_loss = 0.
        train_acc = 0.

        optimizer.zero_grad()

        text, cls = text.to(self.device), cls.to(self.device)
        output = self.model(text)

        loss = criterion(output, cls)

        loss.backward()

        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
            self.model.fc.weight.div_(torch.norm(self.model.fc.weight, dim=1, keepdim=True))

        optimizer.step()

        train_loss += loss.item()
        train_acc += (output.argmax(1) == cls).sum().item()

        del text, cls, output

        return train_loss, train_acc

    def validate(self, validation_set):
        """
        Parameters
        ----------
        validation_set : active_learning.integrations.pytorch.PytorchTextClassificationDataset
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

        valid_iter = dataloader(validation_set, self.mini_batch_size, self._create_collate_fn(),
                                train=False)

        for x, cls in valid_iter:
            x, cls = x.to(self.device), cls.to(self.device)

            with torch.no_grad():
                output = self.model(x)
                loss = self.criterion(output, cls)
                valid_loss += loss.item()
                acc += (output.argmax(1) == cls).sum().item()
                del output, x, cls

        return valid_loss / len(validation_set), acc / len(validation_set)

    def predict(self, test_set, return_proba=False):
        """
        Parameters
        ----------
        test_set : active_learning.integrations.pytorch.PytorchTextClassificationDataset
            Test set.
        """

        proba = self.predict_proba(test_set)
        predictions = np.argmax(proba, axis=1)

        if return_proba:
            return predictions, proba

        return predictions

    def predict_proba(self, test_set):
        self.model.eval()
        test_iter = dataloader(test_set, self.mini_batch_size, self._create_collate_fn(),
                               train=False)

        predictions = []

        with torch.no_grad():
            for text, _ in test_iter:
                text = text.to(self.device)

                predictions += F.softmax(self.model.forward(text), dim=1).to('cpu').tolist()

                del text

        return np.array(predictions)

    def __del__(self):
        try:
            del self.criterion, self.optimizer, self.scheduler, self.model
        except:
            pass
