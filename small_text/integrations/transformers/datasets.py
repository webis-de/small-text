import numpy as np

from small_text.base import LABEL_UNLABELED
from small_text.data.datasets import check_size, get_updated_target_labels
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.labels import list_to_csr

try:
    import torch

    from small_text.integrations.pytorch.datasets import (
        PytorchDataset, PytorchDatasetView
    )
except ModuleNotFoundError:
    raise PytorchNotFoundError('Could not import torchtext')


class TransformersDatasetView(PytorchDatasetView):

    def clone(self):
        if self.is_multi_label:
            data = [(torch.clone(d[TransformersDataset.INDEX_TEXT]),
                     torch.clone(d[TransformersDataset.INDEX_MASK]),
                     d[TransformersDataset.INDEX_LABEL].copy()) for d in self.data]
        else:
            data = [(torch.clone(d[TransformersDataset.INDEX_TEXT]),
                     torch.clone(d[TransformersDataset.INDEX_MASK]),
                     np.copy(d[TransformersDataset.INDEX_LABEL])) for d in self.data]
        return TransformersDataset(data,
                                   multi_label=self.is_multi_label,
                                   target_labels=np.copy(self.target_labels))


class TransformersDataset(PytorchDataset):
    """Dataset class for classifiers from Transformers Integration.
    """
    INDEX_TEXT = 0
    INDEX_MASK = 1
    INDEX_LABEL = 2

    NO_LABEL = -1

    def __init__(self, data, multi_label=False, target_labels=None):
        """
        Parameters
        ----------
        data : list of 3-tuples (text data [Tensor], mask [Tensor], labels [int or list of int])
            The single items constituting the dataset. For single-label datasets, unlabeled
            instances the label should be set to small_text.base.LABEL_UNLABELED`,
            and for multi-label datasets to an empty list.
        multi_label : bool, default=False
            Indicates if this is a multi-label dataset.
        target_labels : numpy.ndarray[int] or None, default=None
            This is a list of (integer) labels to be encountered within this dataset.
            This is important to set if your data does not contain some labels,
            e.g. due to dataset splits, where the labels should however be considered by
            entities such as the classifier. If `None`, the target labels will be inferred
            from the labels encountered in `self.data`.
        """
        self._data = data
        self.multi_label = multi_label

        if target_labels is not None:
            self.track_target_labels = False
            self._target_labels = np.array(target_labels)
        else:
            self.track_target_labels = True
            self._infer_target_labels()

    def _infer_target_labels(self):
        inferred_target_labels = self._get_flattened_unique_labels()
        self.target_labels = inferred_target_labels

    def _get_flattened_unique_labels(self):
        if self.multi_label:
            labels = np.concatenate([d[self.INDEX_LABEL] for d in self._data
                                     if d[self.INDEX_LABEL] is not None
                                     and len(d[self.INDEX_LABEL].shape) > 0])
            labels = np.unique(labels)
        else:
            labels = np.unique([d[self.INDEX_LABEL] for d in self._data])
        return labels

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
            self._data[i] = (_x, self._data[i][self.INDEX_MASK], self._data[i][self.INDEX_LABEL])

    @property
    def y(self):
        if self.multi_label:
            label_list = [d[self.INDEX_LABEL] if d[self.INDEX_LABEL] is not None else []
                          for d in self._data]
            return list_to_csr(label_list, shape=(len(self.data), len(self._target_labels)))
        else:
            return np.array([d[self.INDEX_LABEL] if d[self.INDEX_LABEL] is not None
                             else LABEL_UNLABELED
                             for d in self._data], dtype=int)

    @y.setter
    def y(self, y):
        expected_num_samples = len(self.data)
        if self.multi_label:
            num_samples = y.indptr.shape[0] - 1
            check_size(expected_num_samples, num_samples)

            for i, p in enumerate(range(num_samples)):
                self._data[i] = (self._data[i][self.INDEX_TEXT],
                                 self._data[i][self.INDEX_MASK],
                                 y.indices[y.indptr[p]:y.indptr[p+1]])
        else:
            num_samples = y.shape[0]
            check_size(expected_num_samples, num_samples)

            for i, _y in enumerate(y):
                self._data[i] = (self._data[i][self.INDEX_TEXT],
                                 self._data[i][self.INDEX_MASK],
                                 _y)
        self.target_labels = get_updated_target_labels(self.is_multi_label, y, self.target_labels)

    @property
    def data(self):
        return self._data

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
        encountered_labels = self._get_flattened_unique_labels()
        if np.setdiff1d(encountered_labels, target_labels).shape[0] > 0:
            raise ValueError('Cannot remove existing labels from target_labels as long as they '
                             'still exists in the data. Create a new dataset instead.')
        self._target_labels = target_labels

    def clone(self):
        if self.is_multi_label:
            data = [(torch.clone(d[self.INDEX_TEXT]),
                     torch.clone(d[self.INDEX_MASK]),
                     d[self.INDEX_LABEL].copy()) for d in self._data]
        else:
            data = [(torch.clone(d[self.INDEX_TEXT]),
                     torch.clone(d[self.INDEX_MASK]),
                     np.copy(d[self.INDEX_LABEL])) for d in self._data]
        return TransformersDataset(data,
                                   multi_label=self.multi_label,
                                   target_labels=np.copy(self._target_labels))

    def to(self, other, non_blocking=False, copy=False):

        data = [(d[self.INDEX_TEXT].to(other, non_blocking=non_blocking, copy=copy),
                 d[self.INDEX_MASK].to(other, non_blocking=non_blocking, copy=copy),
                 d[self.INDEX_LABEL]) for d in self._data]

        if copy is True:
            target_labels = None if self.track_target_labels else self._target_labels
            return TransformersDataset(data, target_labels=target_labels)
        else:
            self._data = data
            return self

    def __getitem__(self, item):
        return TransformersDatasetView(self, item)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)
