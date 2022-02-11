import numpy as np

from small_text.base import LABEL_UNLABELED
from small_text.data import Dataset, DatasetView
from small_text.data.exceptions import UnsupportedOperationException
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.labels import list_to_csr

try:
    import torch
except ModuleNotFoundError:
    raise PytorchNotFoundError('Could not import torchtext')


class PytorchDataset(Dataset):

    def __init__(self, device=None):
        self.device = device

    def to(self, other, non_blocking=False, copy=False):
        raise NotImplementedError()

    def to(self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format):
        raise NotImplementedError()


class PytorchDatasetView(DatasetView):

    def __init__(self, dataset, selection):
        self.obj_class = type(self)
        self._dataset = dataset

        self.selection = selection

    @property
    def x(self):
        """Returns the features.

        Returns
        -------
        x :
        """
        selection = self.selection
        if isinstance(self.selection, slice):
            indices = np.arange(len(self._dataset))
            selection = indices[self.selection]
        elif isinstance(self.selection, int):
            selection = [self.selection]

        return [self._dataset.x[i] for i in selection]

    @x.setter
    def x(self, x):
        raise UnsupportedOperationException('Cannot set x on a DatasetView')

    @property
    def is_multi_label(self):
        return self._dataset.is_multi_label

    @property
    def data(self):
        selection = self.selection
        if isinstance(self.selection, slice):
            indices = np.arange(len(self._dataset))
            selection = indices[self.selection]
        elif isinstance(self.selection, int):
            selection = [self.selection]
        return [self._dataset.data[i] for i in selection]

    @property
    def vocab(self):
        return self._dataset.vocab

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        if isinstance(self.selection, slice):
            indices = np.arange(len(self._dataset))
            return indices[self.selection].shape[0]
        elif isinstance(self.selection, int):
            return 1
        return len(self.selection)


class PytorchTextClassificationDataset(PytorchDataset):
    """
    Dataset class for classifiers from Pytorch Integration.
    """
    INDEX_TEXT = 0
    INDEX_LABEL = 1

    def __init__(self, data, vocab, multi_label=False, target_labels=None, device=None):
        """
        Parameters
        ----------
        data : list of tuples (text data [Tensor], label)
            Data set.
        vocab : torchtext.vocab.vocab
            Vocabulary object.
        """
        # TODO: assert label indices are sorted
        self._data = data
        self._vocab = vocab
        self.multi_label = multi_label

        self._target_labels = None
        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = np.array(target_labels)
        else:
            self.track_target_labels = True
            self._infer_target_labels()

        if device is None:
            self.device = None if len(data) == 0 else data[0][self.INDEX_TEXT].device
        else:
            self.device = device

        super().__init__(device=device)

    def _infer_target_labels(self):
        if self.multi_label:
            # TODO: test "and len(d[self.INDEX_LABEL]) > 0"
            inferred_target_labels = np.concatenate([d[self.INDEX_LABEL] for d in self._data
                                                     if d[self.INDEX_LABEL] is not None
                                                     and len(d[self.INDEX_LABEL]) > 0])
            inferred_target_labels = np.unique(inferred_target_labels)
        else:
            inferred_target_labels = np.unique([d[self.INDEX_LABEL] for d in self._data])
        self.target_labels = inferred_target_labels

    @property
    def x(self):
        return [d[self.INDEX_TEXT] for d in self._data]

    @x.setter
    def x(self, x):
        for i, _x in enumerate(x):
            self._data[i] = (_x, self._data[i][self.INDEX_LABEL])

    @property
    def y(self):
        if self.multi_label:
            label_list = [d[self.INDEX_LABEL] if d[self.INDEX_LABEL] is not None else []
                          for d in self._data]
            return list_to_csr(label_list, shape=(len(self.data), len(self.target_labels)))
        else:
            # TODO: document that None is mapped to -1
            return np.array([d[self.INDEX_LABEL] if d[self.INDEX_LABEL] is not None else LABEL_UNLABELED
                             for d in self._data], dtype=int)

    @y.setter
    def y(self, y):
        # TODO: check same length
        if self.multi_label:
            for i, p in enumerate(range(y.indptr.shape[0])):
                self._data[i] = (self._data[i][self.INDEX_TEXT], y.indices[p:(p+1)])
        else:
            for i, _y in enumerate(y):
                self._data[i] = (self._data[i][self.INDEX_TEXT], _y)
        self._infer_target_labels()

    @property
    def data(self):
        """Returns the internal list of tuples storing the data.

        Returns
        -------
        data : list of tuples (text data [Tensor], label)
            Vocab object.
        """
        return self._data

    @property
    def vocab(self):
        """Returns the vocab.

        Returns
        -------
        vocab : torchtext.vocab.Vocab
            Vocab object.
        """
        return self._vocab

    @property
    def is_multi_label(self):
        return self.multi_label

    @property
    def target_labels(self):
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        # TODO: how to handle existing labels that are outside of this set
        self._target_labels = target_labels

    def to(self, other, non_blocking=False, copy=False):
        """Calls `torch.Tensor.to` on all Tensors in `data`.

        Returns
        -------
        self : PytorchTextClassificationDataset
            The object with `to` having been called on all Tensors in `data`.

        See also
        --------
        `PyTorch Docs - torch.Tensor.to <https://pytorch.org/docs/stable/generated/torch.Tensor.to.html>`_
        """
        data = [(d[self.INDEX_TEXT].to(other, non_blocking=non_blocking, copy=copy),
                 d[self.INDEX_LABEL]) for d in self._data]

        if copy is True:
            target_labels = None if self.track_target_labels else self._target_labels
            # TODO: clone vocab
            vocab = self._vocab
            return PytorchTextClassificationDataset(data, vocab, target_labels=target_labels,
                                                    device=self.device)
        else:
            self._data = data
            return self

    def to(self, device=None, dtype=None, non_blocking=False, copy=False,
           memory_format=torch.preserve_format):
        """Calls `torch.Tensor.to` on all Tensors in `data`.

        Returns
        -------
        self : PytorchTextClassificationDataset
            The object with `to` having been called on all Tensors in `data`.

        See also
        --------
        `PyTorch Docs - torch.Tensor.to <https://pytorch.org/docs/stable/generated/torch.Tensor.to.html>`_
        """
        data = [(d[self.INDEX_TEXT].to(device=device, dtype=dtype, non_blocking=non_blocking,
                                       copy=copy, memory_format=memory_format),
                 d[self.INDEX_LABEL]) for d in self._data]

        if copy is True:
            target_labels = None if self.track_target_labels else self._target_labels
            # TODO: clone vocab
            vocab = self._vocab
            return PytorchTextClassificationDataset(data, vocab, target_labels=target_labels,
                                                    device=device)
        else:
            self._data = data
            return self

    def __getitem__(self, item):
        return PytorchDatasetView(self, item)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)
