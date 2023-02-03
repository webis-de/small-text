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


def _check_train_kwargs(train_kwargs):
    if 'max_length' in train_kwargs:
        raise ValueError('Invalid keyword argument in setfit_train_kwargs: '
                         'Argument "max_length" can be set via "max_seq_len" in '
                         'SetFitClassification.')
    return train_kwargs


def _truncate_texts(setfit_model, max_seq_len, *datasets):
    tokenizer = setfit_model.model_body.tokenizer
    datasets_out = []
    for dataset in datasets:
        token_list = [tokenizer.encode(text, verbose=False) for text in dataset.x]
        if any([len(tokens) > max_seq_len for tokens in token_list]):
            x_new = [
                tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(
                        tokenizer.encode(text, max_length=max_seq_len, truncation=True, add_special_tokens=False)
                    )
                )
                for text in dataset.x
            ]
            dataset_copy = dataset.clone()
            dataset_copy.x = x_new
            datasets_out.append(dataset_copy)
        else:
            datasets_out.append(dataset)
    return datasets_out
