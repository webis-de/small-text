import warnings
import numpy as np

from packaging.version import parse, Version
from typing import TYPE_CHECKING, Union

from numpy.random import choice
from sklearn.cluster import kmeans_plusplus

# if TYPE_CHECKING:
import numpy.typing as npt
from numpy.random import RandomState


def init_kmeans_plusplus_safe(x: npt.NDArray[np.double],
                              n_clusters: int,
                              *,
                              weights: Union[npt.NDArray[np.double], None] = None,
                              x_squared_norms: Union[npt.NDArray[np.double], None] = None,
                              random_state: Union[int, RandomState, None] = None,
                              n_local_trials: Union[int, None] = None,
                              suppress_warning: bool = False):
    """
    Calls scikit-learn's k-means++ initialization with a fallback that prevents duplicate
    cluster centers. If duplicate indices are encountered, they are replaced with randomly sampled
    indices and a warning is raised.

    Parameters
    ----------
    x : numpy.ndarray[double]
        The input data to be clustered.
    n_clusters : int
        The number of cluster centers to find.
    weights : numpy.ndarray[double], default=None

        Sample weights or None. Using this requires scikit-learn >= 1.3.0.

        .. versionadded:: 2.0.0

    x_squared_norms : numpy.ndarray[double], default=None
        List of squared norms or None.
    random_state : int or RandomState instance or None, default=None
        A random state or None.
    n_local_trials : int, default=None
        Number of local trials.
    suppress_warning : bool, default=None
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
    kmeans_plusplus_kwargs = {}
    if weights is not None:
        from sklearn import __version__
        if parse(__version__) < Version('1.3.0'):
            raise ValueError('scikit-learn>=1.3.0 is required to use the "weights" argument.')

        kmeans_plusplus_kwargs['sample_weight'] = weights

    centers, indices = kmeans_plusplus(x,
                                       n_clusters,
                                       x_squared_norms=x_squared_norms,
                                       random_state=random_state,
                                       n_local_trials=n_local_trials,
                                       **kmeans_plusplus_kwargs)

    unique_indices, counts = np.unique(indices, return_counts=True)

    if unique_indices.shape[0] != n_clusters:
        if not suppress_warning:
            warnings.warn('kmeans_plusplus returned identical cluster centers.')

        remaining_indices = np.arange(x.shape[0])
        remaining_indices = np.delete(remaining_indices, unique_indices)

        centers = np.delete(centers, np.arange(unique_indices.shape[0])[counts > 1], axis=0)

        fill_indices = choice(remaining_indices, size=n_clusters-unique_indices.shape[0], replace=False)

        indices = np.hstack((unique_indices, fill_indices))
        centers = np.vstack((centers, x[fill_indices]))

        return centers, indices

    return centers, indices
