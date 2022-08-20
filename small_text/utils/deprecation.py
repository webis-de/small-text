import inspect
import warnings

from functools import wraps
from packaging import version

from small_text.version import get_version


class DeprecationError(ValueError):
    """Thrown when the current version is greater than the planned time of deprecation."""
    pass


def deprecated(func_or_class=None, deprecated_in=None, to_be_removed_in=None, replacement=None):
    if deprecated_in is None:
        raise ValueError('Keyword argument \'deprecated_in\' must be set.')

    if func_or_class is not None:
        if not inspect.isclass(func_or_class) and not inspect.isfunction(func_or_class):
            raise ValueError('The @deprecated decorator requires a function or class')

    def _decorator(func_or_class):
        subject = 'class' if inspect.isclass(func_or_class) else 'function'

        deprecation_version = version.parse(deprecated_in)
        removal_version = version.parse(to_be_removed_in) if to_be_removed_in is not None else None
        current_version = get_version()

        if removal_version is not None and current_version >= removal_version:
            raise DeprecationError(f'The {subject} {func_or_class.__name__} should have been '
                                   f'removed before version {str(current_version)}.')

        @wraps(func_or_class)
        def wrapper(*args, **kwargs):
            removed_in = f' and will be removed in {str(removal_version)}' \
                if removal_version is not None else ''
            full_stop = '.' if replacement is None else ''
            replacement_text = f'. Please use {replacement} instead.' if replacement is not None else ''

            warnings.warn(f'The {subject} {func_or_class.__name__} has been deprecated in '
                          f'{str(deprecation_version)}'
                          f'{removed_in}{full_stop}{replacement_text}',
                          DeprecationWarning,
                          stacklevel=2)
            return func_or_class(*args, **kwargs)
        return wrapper
    return _decorator(func_or_class) if callable(func_or_class) else _decorator


def early_stopping_deprecation_warning(early_stopping_no_improvement, early_stopping_acc):
    if early_stopping_no_improvement != 5:
        warnings.warn(
            'early_stopping_no_improvement is deprecated since 1.1.0 '
            'and will be removed in 2.0.0. Please use the "early_stopping" kwarg in fit () '
            'instead.',
            DeprecationWarning,
            stacklevel=2)
    if early_stopping_acc != -1:
        warnings.warn(
            'early_stopping_acc  is deprecated since 1.1.0 '
            'and will be removed in 2.0.0. Please use the "early_stopping" kwarg in fit () '
            'instead.',
            DeprecationWarning,
            stacklevel=2)


def model_selection_deprecation_warning(model_selection):
    if model_selection is not True:
        warnings.warn(
            'model_selection is deprecated since 1.1.0 '
            'and will be removed in 2.0.0. Please use the "model_selection" kwarg in fit () '
            'instead.',
            DeprecationWarning,
            stacklevel=2)
