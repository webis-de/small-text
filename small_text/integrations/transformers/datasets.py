import numpy as np

from scipy.sparse import csr_matrix
from small_text.base import LABEL_UNLABELED
from small_text.data.datasets import check_size, check_target_labels, get_updated_target_labels
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.annotations import experimental
from small_text.utils.labels import csr_to_list, get_num_labels, list_to_csr

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

        target_labels = None if self.dataset.track_target_labels else np.copy(self.target_labels)
        return TransformersDataset(data,
                                   multi_label=self.is_multi_label,
                                   target_labels=target_labels)


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
        check_target_labels(target_labels)

        self._data = data
        self.multi_label = multi_label

        if target_labels is not None:
            self.track_target_labels = False
            self._target_labels = np.array(target_labels)
        else:
            self.track_target_labels = True
            self._infer_target_labels()

    def _infer_target_labels(self):
        if len(self._data) == 0:
            self.target_labels = np.array([0])
        else:
            unique_labels = np.unique(self._get_flattened_labels())
            if unique_labels.shape[0] > 0:
                max_label_id = unique_labels.max()
                self.target_labels = np.arange(max_label_id + 1)
            else:
                self.target_labels = np.array([0])

    def _get_flattened_labels(self):
        label_list = [d[self.INDEX_LABEL] if d[self.INDEX_LABEL] is not None else []
                      for d in self._data]
        if self.multi_label:
            label_list = [label for lst in label_list for label in lst]

        label_list = [label for label in label_list if label > LABEL_UNLABELED]

        return label_list

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
            if self.track_target_labels:
                # TODO: int() cast should not be necessary here
                num_classes = int(self.target_labels.max()) + 1
            else:
                num_classes = len(self.target_labels)
            return list_to_csr(label_list, shape=(len(self.data), num_classes))
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

        if self.track_target_labels:
            self.target_labels = get_updated_target_labels(self.is_multi_label,
                                                           y,
                                                           self.target_labels)
        else:
            max_label_id = get_num_labels(y) - 1
            max_target_labels_id = self.target_labels.max()
            if max_label_id > max_target_labels_id:
                raise ValueError(f'Error while assigning new labels to dataset: '
                                 f'Encountered label with id {max_label_id} which is outside of '
                                 f'the configured set of target labels (whose maximum label is '
                                 f'is {max_target_labels_id}) [track_target_labels=False]')

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
        encountered_labels = np.unique(self._get_flattened_labels())
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

        target_labels = None if self.track_target_labels else np.copy(self._target_labels)
        return TransformersDataset(data,
                                   multi_label=self.multi_label,
                                   target_labels=target_labels)

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

    @classmethod
    @experimental
    def from_arrays(cls, texts, y, tokenizer, target_labels=None, max_length=512):
        """Constructs a new TransformersDataset from the given text and label arrays.

        Parameters
        ----------
        texts : list of str or np.ndarray[str]
            List of text documents.
        y : np.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
            Depending on the type of `y` the resulting dataset will be single-label (`np.ndarray`)
            or multi-label (`scipy.sparse.csr_matrix`).
        tokenizer : tokenizers.Tokenizer
            A huggingface tokenizer.
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be directly passed to the datset constructor.
        max_length : int
            Maximum sequence length.
        train : bool
            If `True` fits the vectorizer and transforms the data, otherwise just transforms the
            data.

        Returns
        -------
        dataset : TransformersDataset
            A dataset constructed from the given texts and labels.


        .. warning::
           This functionality is still experimental and may be subject to change.

        .. versionadded:: 1.1.0
        """
        data_out = []

        multi_label = isinstance(y, csr_matrix)
        if multi_label:
            y = csr_to_list(y)

        for i, doc in enumerate(texts):
            encoded_dict = tokenizer.encode_plus(
                doc,
                add_special_tokens=True,
                padding='max_length',
                max_length=max_length,
                return_attention_mask=True,
                return_tensors='pt',
                truncation='longest_first'
            )

            if multi_label:
                data_out.append((encoded_dict['input_ids'],
                                 encoded_dict['attention_mask'],
                                 np.sort(y[i])))
            else:
                data_out.append((encoded_dict['input_ids'],
                                 encoded_dict['attention_mask'],
                                 y[i]))

        return TransformersDataset(data_out, multi_label=multi_label, target_labels=target_labels)

    def __getitem__(self, item):
        return TransformersDatasetView(self, item)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)
