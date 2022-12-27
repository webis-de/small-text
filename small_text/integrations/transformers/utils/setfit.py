def _check_model_kwargs(model_kwargs):
    if 'use_differentiable_head' in model_kwargs:
        raise ValueError('Invalid keyword argument in model_kwargs: '
                         'Argument "use_differentiable_head" is managed by '
                         'SetFitClassification.')

    if 'force_download' in model_kwargs:
        raise ValueError('Invalid keyword argument in model_kwargs: '
                         'Argument "force_download" is managed by '
                         'SetFitClassification via the argument setfit_model_args.')

    if 'local_files_only' in model_kwargs:
        raise ValueError('Invalid keyword argument in model_kwargs: '
                         'Argument "local_files_only" is managed by '
                         'SetFitClassification via the argument setfit_model_args.')

    return model_kwargs


def _check_trainer_kwargs(trainer_kwargs):
    if 'batch_size' in trainer_kwargs:
        raise ValueError('Invalid keyword argument in trainer_kwargs: '
                         'Argument "batch_size" can be set via "mini_batch_size" in '
                         'SetFitClassification.')
    return trainer_kwargs
