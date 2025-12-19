import os

from transformers import logging as transformers_logging
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from small_text.integrations.transformers.classifiers.base import (
    ModelLoadingStrategy,
    PretrainedModelLoadingArguments
)


MANAGED_CONFIG_KWARGS = set(['num_labels', 'cache_dir', 'force_download'])

MANAGED_TOKENIZER_KWARGS = set(['cache_dir', 'force_download'])

MANAGED_MODEL_KWARGS = set(['from_tf', 'config', 'cache_dir', 'force_download', 'local_files_only'])


def _check_for_managed_config_kwargs(encountered_kwargs):
    for kwarg in encountered_kwargs.keys():
        if kwarg in MANAGED_CONFIG_KWARGS:
            _raise_managed_kwargs_error('config', kwarg, MANAGED_CONFIG_KWARGS)
    return encountered_kwargs


def _check_for_managed_tokenizer_kwargs(encountered_kwargs):
    for kwarg in encountered_kwargs.keys():
        if kwarg in MANAGED_TOKENIZER_KWARGS:
            _raise_managed_kwargs_error('tokenizer', kwarg, MANAGED_TOKENIZER_KWARGS)
    return encountered_kwargs


def _check_for_managed_model_kwargs(encountered_kwargs):
    for kwarg in encountered_kwargs.keys():
        if kwarg in MANAGED_MODEL_KWARGS:
            _raise_managed_kwargs_error('model', kwarg, MANAGED_MODEL_KWARGS)
    return encountered_kwargs


def _raise_managed_kwargs_error(managed_kwargs_type, kwargs, managed_kwargs):
    raise ValueError(f'Cannot override managed keyword argument in {managed_kwargs_type}_kwargs: "{kwargs}". '
                     f'Managed keyword arguments: {list(managed_kwargs)}')


def _get_arguments_for_from_pretrained_model(model_loading_strategy: ModelLoadingStrategy) \
        -> PretrainedModelLoadingArguments:

    if model_loading_strategy == ModelLoadingStrategy.DEFAULT:
        if str(os.environ.get('TRANSFORMERS_OFFLINE', '0')) == '1':
            # same as ALWAYS_LOCAL
            return PretrainedModelLoadingArguments(local_files_only=True)
        else:
            return PretrainedModelLoadingArguments()
    elif model_loading_strategy == ModelLoadingStrategy.ALWAYS_LOCAL:
        return PretrainedModelLoadingArguments(local_files_only=True)
    else:
        return PretrainedModelLoadingArguments(force_download=True)


def _initialize_transformer_components(transformer_model,
                                       num_classes: int,
                                       cache_dir: str):

    from_pretrained_options = _get_arguments_for_from_pretrained_model(
        transformer_model.model_loading_strategy
    )

    config = AutoConfig.from_pretrained(
        transformer_model.config,
        num_labels=num_classes,
        cache_dir=cache_dir,
        force_download=from_pretrained_options.force_download,
        **transformer_model.config_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        transformer_model.tokenizer,
        cache_dir=cache_dir,
        force_download=from_pretrained_options.force_download,
        **transformer_model.tokenizer_kwargs
    )

    # Suppress "Some weights of the model checkpoint at [model name] were not [...]"-warnings
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    model = AutoModelForSequenceClassification.from_pretrained(
        transformer_model.model,
        from_tf=False,
        config=config,
        cache_dir=cache_dir,
        force_download=from_pretrained_options.force_download,
        local_files_only=from_pretrained_options.local_files_only,
        **transformer_model.model_kwargs
    )
    transformers_logging.set_verbosity(previous_verbosity)

    return config, tokenizer, model

