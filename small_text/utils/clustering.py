import warnings
import numpy as np

from numpy.random import choice
from sklearn.cluster import kmeans_plusplus


def init_kmeans_plusplus_safe(x, n_clusters, *, x_squared_norms=None, random_state=None,
                              n_local_trials=None, suppress_warning=False):
    """
    Calls scikit-learn's k-means++ initialization with a fallback that prevents duplicate
    cluster centers. If duplicate indices are encountered, they are replaced with randomly sampled
    indices and a warning is raised.

    Parameters
    ----------
    x : numpy.ndarray
        The input data to be clustered.
    n_clusters : int
        The number of cluster centers to find.
    x_squared_norms : numpy.ndarray
        List of squared norms or None.
    random_state : int or RandomState instance
        A random state or None.
    n_local_trials : int
        Number of local trials.
    suppress_warning : bool
        Suppresses the warning given on duplicate indices if True.

    Returns
    -------
    centers : numpy.ndarray
        `n_clusters`-many cluster centers.
    indices :
        Indices (relativ to `x`) of the returned cluster centers.

    See Also
    --------
    sklearn.cluster import kmeans_plusplus :
        The actual k-means++ initialization provided by scikit-learn.
    """

    centers, indices = kmeans_plusplus(x,
                                       n_clusters,
                                       x_squared_norms=x_squared_norms,
                                       random_state=random_state,
                                       n_local_trials=n_local_trials)

    unique_indices, counts = np.unique(indices, return_counts=True)

    if unique_indices.shape[0] != n_clusters:
        if not suppress_warning:
            warnings.warn('kmeans_plusplus returned identical cluster centers.')

        remaining_indices = np.arange(x.shape[0])
        remaining_indices = np.delete(remaining_indices, unique_indices)

        centers = np.delete(centers, np.arange(unique_indices.shape[0])[counts > 1], axis=0)

        fill_indices = choice(remaining_indices, size=n_clusters-unique_indices.shape[0],
                              replace=False)
        indices = np.hstack((unique_indices, fill_indices))
        centers = np.vstack((centers, x[fill_indices]))

        return centers, indices

    return centers, indices
