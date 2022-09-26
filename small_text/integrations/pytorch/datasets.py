import torch
import numpy as np

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from small_text.base import LABEL_UNLABELED
from small_text.data import DatasetView
from small_text.data.datasets import check_size, check_target_labels, get_updated_target_labels
from small_text.data.exceptions import UnsupportedOperationException
from small_text.utils.annotations import experimental
from small_text.utils.labels import csr_to_list, get_num_labels, list_to_csr


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

        dataset = self
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        if dataset.track_target_labels:
            target_labels = None
        else:
            target_labels = np.copy(dataset.target_labels)

        return PytorchTextClassificationDataset(data,
                                                copy.deepcopy(self.vocab),
                                                multi_label=self.is_multi_label,
                                                target_labels=target_labels)


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
        check_target_labels(target_labels)

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
            self._data[i] = (_x, self._data[i][self.INDEX_LABEL])

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

        if self.track_target_labels:
            self.target_labels = get_updated_target_labels(self.is_multi_label, y, self.target_labels)
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
        encountered_labels = np.unique(self._get_flattened_labels())
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

        if self.track_target_labels:
            target_labels = None
        else:
            target_labels = np.copy(self._target_labels)

        return PytorchTextClassificationDataset(data,
                                                copy.deepcopy(self._vocab),
                                                multi_label=self.multi_label,
                                                target_labels=target_labels)

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

    @classmethod
    @experimental
    def from_arrays(cls, texts, y, text_field, target_labels=None, train=True):
        """Constructs a new PytorchTextClassificationDataset from the given text and label arrays.

        Parameters
        ----------
        texts : list of str or np.ndarray[str]
            List of text documents.
        y : np.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
            Depending on the type of `y` the resulting dataset will be single-label (`np.ndarray`)
            or multi-label (`scipy.sparse.csr_matrix`).
        text_field : torchtext.data.field.Field or torchtext.legacy.data.field.Field
            A torchtext field used for preprocessing the text and building the vocabulary.
        vocab : object
            A torch
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be directly passed to the datset constructor.
        train : bool
            If `True` fits the vectorizer and transforms the data, otherwise just transforms the
            data.

        Returns
        -------
        dataset : PytorchTextClassificationDataset
            A dataset constructed from the given texts and labels.


        .. warning::
           This functionality is still experimental and may be subject to change.

        .. versionadded:: 1.1.0
        """
        unk_token_idx = 0  # TODO: check this index

        if not train and not hasattr(text_field, 'vocab'):
            raise ValueError('Vocab must have been built when using this function '
                             'to obtain test data.')

        texts_preprocessed = [text_field.preprocess(text) for text in texts]

        if train:
            text_field.build_vocab(texts_preprocessed)
            assert text_field.vocab.itos[0] == '<unk>'
            assert text_field.vocab.itos[1] == '<pad>'

        multi_label = isinstance(y, csr_matrix)
        if multi_label:
            y = csr_to_list(y)

        vocab = text_field.vocab

        data = [
            (
                torch.LongTensor([vocab.stoi[token] if token in vocab.stoi else unk_token_idx
                                 for token in text_preprocessed]),
                y[i]
            )
            for i, text_preprocessed in enumerate(texts_preprocessed)
        ]

        return PytorchTextClassificationDataset(data, vocab, multi_label=multi_label,
                                                target_labels=target_labels)

    def __getitem__(self, item):
        return PytorchTextClassificationDatasetView(self, item)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)
