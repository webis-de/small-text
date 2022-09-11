import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


def multilabel_stratified_subsets_sampling(y, n_samples=10):
    """Provides stratified sampling for multi-label data using the subsets sampling
    approach [STV11]_.

    Parameters
    ----------
    y : scipy.sparse.csr_matrix
        List of labels.
    n_samples : int, default=10
        Number of indices to sample.

    Returns
    -------
    indices : numpy.ndarray
        Indices of the stratified subset.

    References
    ----------
    .. [STV11] Konstantinos Sechidis, Grigorios Tsoumakas, and Ioannis Vlahavas.
       On the stratification of multi-label data.
       In: Proceedings of the 2011 European Conference on Machine Learning and Knowledge Discovery
          in Databases - Volume Part III (ECML PKDD'11).
       Springer-Verlag, Berlin, Heidelberg, 145--158.
    """
    y = y.toarray()
    y = np.apply_along_axis(np.array2string, 1, y)

    le = LabelEncoder()
    y_labelsets = le.fit_transform(y)

    return stratified_sampling(y_labelsets, n_samples=n_samples)


def stratified_sampling(y, n_samples=10, enforce_min_occurrence=True):
    """
    Performs a stratified random sampling.

    Parameters
    ----------
    y : numpy.ndarray or scipy.sparse.csr_matrix
        Dense or sparse matrix of of labels.
    n_samples : int, default=10
        Number of indices to sample.
    enforce_min_occurrence : bool, default=True
        Ensures that at least one sample from each class (provided it is present in the data)
        is included in the stratified sample.

    Returns
    -------
    indices : numpy.ndarray
        Indices of the stratified subset.

    Notes
    -----
    Only useful for experimental simulations (Requires label knowledge).
    """
    _assert_sample_size(y, n_samples)

    # TODO: check for gaps in given labels

    # num classes according to the labels
    num_classes = np.max(y) + 1

    counts = _get_class_histogram(y, num_classes)
    expected_samples_per_class = np.floor(counts * (float(n_samples) / counts.sum())).astype(int)

    if enforce_min_occurrence and expected_samples_per_class.min() == 0:
        if n_samples > num_classes and np.unique(y).shape[0] == num_classes:  # feasibility check
            expected_samples_per_class += 1

            num_excessive_samples = expected_samples_per_class.sum() - n_samples
            class_indices = np.arange(counts.shape[0])[expected_samples_per_class > 1]
            round_robin_index = 0
            for i in range(num_excessive_samples):

                while expected_samples_per_class[class_indices[round_robin_index]] <= 1:
                    round_robin_index += 1
                    round_robin_index %= class_indices.shape[0]

                expected_samples_per_class[class_indices[round_robin_index]] -= 1

                class_indices = np.arange(counts.shape[0])[expected_samples_per_class > 1]
                assert expected_samples_per_class[class_indices].sum() > 0

    return _random_sampling(n_samples, num_classes, expected_samples_per_class, counts, y)


def balanced_sampling(y, n_samples=10):
    """
    Performs a class-balanced random sampling.

    If `n_samples` is not divisible by the number of classes, a number of samples equal to the
    remainder will be sampled randomly among the classes.

    Parameters
    ----------
    y : list of int or numpy.ndarray
        List of labels.
    n_samples : int, default=10
        Number of indices to sample.

    Returns
    -------
    indices : numpy.ndarray
        Indices of the stratified subset.

    Notes
    -----
    Only useful for experimental simulations (Requires label knowledge).
    """
    _assert_sample_size(y, n_samples)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # TODO: check for gaps in given labels

    # num classes according to the labels
    num_classes = np.max(y) + 1
    # num classes encountered
    num_classes_present = len(np.unique(y))

    counts = _get_class_histogram(y, num_classes)
    expected_samples_per_class = np.array([int(n_samples / num_classes_present)] * num_classes)

    return _random_sampling(n_samples, num_classes, expected_samples_per_class, counts, y)


def _assert_sample_size(y, num_samples):
    label_set_size = len(y)
    if num_samples > label_set_size:
        raise ValueError(f'Error! Requested number of samples {num_samples} '
                         f'exceeds label set size ')


def _random_sampling(n_samples, num_classes, expected_samples_per_class, counts, y):

    remainder = n_samples - expected_samples_per_class.sum()

    for i in range(num_classes):
        diff = expected_samples_per_class[i] - counts[i]
        if diff > 0:
            expected_samples_per_class[i] = counts[i]
            remainder += diff

    if remainder != 0:
        classes = np.array([i for i in range(num_classes)])

        counts_remaining = counts - expected_samples_per_class

        for i in range(remainder):
            p_remaining = counts_remaining.astype(float) / counts_remaining.sum()

            class_index = np.random.choice(classes, size=1, replace=False, p=p_remaining)\
                .flatten().tolist()[0]
            expected_samples_per_class[class_index] += 1
            counts_remaining[class_index] -= 1

    indices = []
    for i in range(num_classes):
        if expected_samples_per_class[i] > 0:
            class_indices = np.argwhere(y == i)[:, 0]
            indices += np.random.choice(class_indices, size=expected_samples_per_class[i],
                                        replace=False).flatten().tolist()

    indices = np.random.permutation(indices)

    return indices


def _get_class_histogram(y, num_classes, normalize=False):
    if isinstance(y, csr_matrix):
        ind, counts = np.unique(y.indices, return_counts=True)
    else:
        ind, counts = np.unique(y, return_counts=True)
    ind = set(ind)

    histogram = np.zeros(num_classes)
    for i, c in zip(ind, counts):
        if i in ind:
            histogram[i] = c

    if normalize:
        histogram = histogram / histogram.sum()
        return histogram

    return histogram.astype(int)
