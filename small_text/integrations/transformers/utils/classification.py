import os

from typing import List

from transformers import logging as transformers_logging
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

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


def _build_layer_specific_params(model: PreTrainedModel, base_lr: float, fine_tuning_arguments):
    """Builds an iterable containing layer-specific parameter groups for use with a torch optimizer. These groups
    are separated by layer, by which we mean a logical grouping of neural network layers within the transformer
    as commonly used in literature (e.g., the embedding layer, multiple encoder layers,
    and a final classification layer).

    If enabled through the given `fine_tuning_arguments`, the layer-specific params are adapted so that
    layer-wise gradient decay and/or gradual unfreezing is performed.

    This function makes certain assumptions. For this functionality to work properly, a `PreTrainedModel` model
    with layers similar to `BertForSequenceClassification` is required.
    """

    layers = _collect_layer_specific_params(model)
    if fine_tuning_arguments.gradual_unfreezing >= len(layers):
        raise ValueError('Invalid gradual unfreezing parameters: No trainable layers left.')

    params, start_layer = _adapt_layer_specific_params(layers, base_lr, fine_tuning_arguments)

    _check_for_completeness(model, layers, params, start_layer)
    return params


def _collect_layer_specific_params(model: PreTrainedModel):

    base_model = getattr(model, model.base_model_prefix)

    layers = []
    if hasattr(base_model, 'embeddings'):
        layers.append(base_model.embeddings.parameters())

    if hasattr(base_model, 'encoder'):
        if hasattr(base_model.encoder, 'layer'):
            layers += [layer.parameters() for layer in base_model.encoder.layer]
    else:
        layers += [layer.parameters() for layer in base_model.transformer.layer]

    if hasattr(base_model, 'pooler') and base_model.pooler is not None:
        layers.append(base_model.pooler.parameters())
    if hasattr(model, 'classifier'):
        layers.append(model.classifier.parameters())

    return layers


def _adapt_layer_specific_params(layers: List[dict], base_lr: float, fine_tuning_arguments):

    layerwise_gradient_decay = fine_tuning_arguments.layerwise_gradient_decay
    use_gradual_unfreezing = isinstance(fine_tuning_arguments.gradual_unfreezing, int) and \
        fine_tuning_arguments.gradual_unfreezing > 0

    total_layers = len(layers)
    start_layer = 0 if not use_gradual_unfreezing else max(0, total_layers - fine_tuning_arguments.gradual_unfreezing)
    num_layers = total_layers - start_layer

    params = []
    for i in range(start_layer, total_layers):
        lr = base_lr if not layerwise_gradient_decay else base_lr * layerwise_gradient_decay ** (
                    num_layers - (i + 1 - start_layer))
        for sublayer in layers[i]:
            if sublayer.requires_grad:  # Check whether frozen through pytorch interface
                params.append({
                    'params': sublayer,
                    'lr': lr
                })

    return params, start_layer


def _check_for_completeness(model, layers, params, start_layer: int):
    """Checks whether all trainable parameters have been used.
    """
    must_have_layer_ids = set(id(param) for param in model.parameters() if param.requires_grad)
    included_layer_ids = set([id(param['params']) for param in params])
    excluded_layer_ids = set([id(sublayer) for i in range(0, start_layer) for sublayer in layers[i]])
    found_layer_ids = set.union(included_layer_ids, excluded_layer_ids)

    if len(must_have_layer_ids - found_layer_ids) != 0:
        # Not all layers were found while following naming convention
        raise ValueError(f'Fine-tuning arguments are not supported for transformer model "{type(model).__name__}".')
