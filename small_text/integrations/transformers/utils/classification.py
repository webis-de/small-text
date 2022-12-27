import os

from transformers import logging as transformers_logging
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from small_text.integrations.transformers.classifiers.base import (
    ModelLoadingStrategy,
    PretrainedModelLoadingArguments
)


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
    )

    tokenizer = AutoTokenizer.from_pretrained(
        transformer_model.tokenizer,
        cache_dir=cache_dir,
        force_download=from_pretrained_options.force_download,
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
        local_files_only=from_pretrained_options.local_files_only
    )
    transformers_logging.set_verbosity(previous_verbosity)

    return config, tokenizer, model
