import numpy as np

from abc import ABC, abstractmethod
from small_text.base import LABEL_UNLABELED
from small_text.data import DatasetView
from small_text.data.datasets import check_size, get_updated_target_labels
from small_text.data.exceptions import UnsupportedOperationException
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.labels import list_to_csr

try:
    import torch
    from small_text.integrations.pytorch.utils.labels import get_flattened_unique_labels
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


class PytorchDataset(ABC):

    @abstractmethod
    def to(self, other, non_blocking=False, copy=False):
        pass


class PytorchDatasetView(DatasetView):

    def __init__(self, dataset, selection):
        self.obj_class = type(self)
        self._dataset = dataset

        self.selection = selection

    @property
    def dataset(self):
        return self._dataset

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
    def y(self):
        return self.dataset.y[self.selection]

    @y.setter
    def y(self, y):
        raise UnsupportedOperationException('Cannot set y on a DatasetView')

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

    @property
    def target_labels(self):
        return self.dataset.target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        raise UnsupportedOperationException('Cannot set target_labels on a DatasetView')

    @abstractmethod
    def clone(self):
        pass

    def __getitem__(self, item):
        return self.obj_class(self, item)

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        if isinstance(self.selection, slice):
            indices = np.arange(len(self._dataset))
            return indices[self.selection].shape[0]
        elif isinstance(self.selection, int):
            return 1
        return len(self.selection)


class PytorchTextClassificationDatasetView(PytorchDatasetView):

    def clone(self):
        import copy

        if self.is_multi_label:
            data = [(torch.clone(d[PytorchTextClassificationDataset.INDEX_TEXT]),
                     d[PytorchTextClassificationDataset.INDEX_LABEL].copy()) for d in self.data]
        else:
            data = [(torch.clone(d[PytorchTextClassificationDataset.INDEX_TEXT]),
                     np.copy(d[PytorchTextClassificationDataset.INDEX_LABEL])) for d in self.data]
        return PytorchTextClassificationDataset(data,
                                                copy.deepcopy(self.vocab),
                                                multi_label=self.is_multi_label,
                                                target_labels=np.copy(self.target_labels))


class PytorchTextClassificationDataset(PytorchDataset):
    """Dataset class for classifiers from Pytorch Integration.
    """
    INDEX_TEXT = 0
    INDEX_LABEL = 1

    def __init__(self, data, vocab, multi_label=False, target_labels=None):
        """
        Parameters
        ----------
        data : list of tuples (text data [Tensor], labels [int or list of int])
            The single items constituting the dataset. For single-label datasets, unlabeled
            instances the label should be set to small_text.base.LABEL_UNLABELED`,
            and for multi-label datasets to an empty list.
        vocab : torchtext.vocab.vocab
            Vocabulary object.
        multi_label : bool, default=False
            Indicates if this is a multi-label dataset.
        target_labels : np.ndarray[int] or None, default=None
            This is a list of (integer) labels to be encountered within this dataset.
            This is important to set if your data does not contain some labels,
            e.g. due to dataset splits, where the labels should however be considered by
            entities such as the classifier. If `None`, the target labels will be inferred
            from the labels encountered in `self.data`.
        """
        self._data = data
        self._vocab = vocab
        self.multi_label = multi_label

        self._target_labels = None
        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = target_labels
        else:
            self.track_target_labels = True
            self._infer_target_labels()

    def _infer_target_labels(self):
        inferred_target_labels = get_flattened_unique_labels(self)
        self.target_labels = inferred_target_labels

    @property
    def x(self):
        """Returns the features.

        Returns
        -------
        x : list of Tensor
        """
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
            return np.array([d[self.INDEX_LABEL]
                             if d[self.INDEX_LABEL] is not None
                             else LABEL_UNLABELED
                             for d in self._data], dtype=int)

    @y.setter
    def y(self, y):
        expected_num_samples = len(self.data)
        if self.multi_label:
            num_samples = y.indptr.shape[0] - 1
            check_size(expected_num_samples, num_samples)

            for i, p in enumerate(range(num_samples)):
                self._data[i] = (
                    self._data[i][self.INDEX_TEXT],
                    y.indices[y.indptr[p]:y.indptr[p+1]]
                )
        else:
            num_samples = y.shape[0]
            check_size(expected_num_samples, num_samples)

            for i, _y in enumerate(y):
                self._data[i] = (self._data[i][self.INDEX_TEXT], _y)

        self.target_labels = get_updated_target_labels(self.is_multi_label, y, self.target_labels)

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
        """Returns the target labels.

        Returns
        -------
        target_labels : list of int
            List of target labels.
        """
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        encountered_labels = get_flattened_unique_labels(self)
        if np.setdiff1d(encountered_labels, target_labels).shape[0] > 0:
            raise ValueError('Cannot remove existing labels from target_labels as long as they '
                             'still exists in the data. Create a new dataset instead.')
        self._target_labels = target_labels

    def clone(self):
        import copy

        if self.is_multi_label:
            data = [(torch.clone(d[self.INDEX_TEXT]),
                     d[self.INDEX_LABEL].copy()) for d in self._data]
        else:
            data = [(torch.clone(d[self.INDEX_TEXT]),
                     np.copy(d[self.INDEX_LABEL])) for d in self._data]
        return PytorchTextClassificationDataset(data,
                                                copy.deepcopy(self._vocab),
                                                multi_label=self.multi_label,
                                                target_labels=np.copy(self._target_labels))

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
            import copy
            target_labels = None if self.track_target_labels else self._target_labels
            vocab = copy.deepcopy(self._vocab)
            return PytorchTextClassificationDataset(data, vocab, target_labels=target_labels)
        else:
            self._data = data
            return self

    def __getitem__(self, item):
        return PytorchTextClassificationDatasetView(self, item)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)
