import numpy as np

from small_text.base import LABEL_UNLABELED


def get_flattened_unique_labels(dataset):
    if dataset.is_multi_label:
        labels = [
            d[dataset.INDEX_LABEL] for d in dataset.data
            if d[dataset.INDEX_LABEL] is not None and len(d[dataset.INDEX_LABEL]) > 0
        ]
        if len(labels) == 0:
            labels = np.array([], dtype=int)
        else:
            labels = np.concatenate([d[dataset.INDEX_LABEL] for d in dataset.data
                                     if d[dataset.INDEX_LABEL] is not None
                                     and len(d[dataset.INDEX_LABEL]) > 0])
            labels = np.unique(labels)
    else:
        labels = np.unique([d[dataset.INDEX_LABEL] for d in dataset._data])
    return np.setdiff1d(labels, np.array([LABEL_UNLABELED]))
