import warnings


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
