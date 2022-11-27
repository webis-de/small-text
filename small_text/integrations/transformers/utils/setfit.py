def _check_model_kwargs(model_kwargs):
    if 'use_differentiable_head' in model_kwargs:
        raise ValueError('Invalid keyword argument in model_kwargs: '
                         'Argument "use_differentiable_head" is managed by '
                         'SetFitClassification.')
    return model_kwargs


def _check_trainer_kwargs(trainer_kwargs):
    if 'batch_size' in trainer_kwargs:
        raise ValueError('Invalid keyword argument in trainer_kwargs: '
                         'Argument "batch_size" can be set via "mini_batch_size" in '
                         'SetFitClassification.')
    return trainer_kwargs
