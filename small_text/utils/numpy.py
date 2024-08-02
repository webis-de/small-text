import numpy as np

from numpy.typing import DTypeLike

from packaging.version import parse, Version


def int_dtype() -> DTypeLike:
    """Returns the default unsigned int dtype for numpy.

    This is used to handle the missing functionality of numpy.intp in numpy versions prior to `2.0.0`.

    .. seealso:: `numpy/numpy#9464 <https://github.com/numpy/numpy/issues/9464>`_

    """
    return np.uintp if parse(np.__version__) >= Version('2.0.0') else np.uint32
