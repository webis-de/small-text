import torch
import numpy as np

from small_text.integrations.transformers.datasets import TransformersDataset


def preprocess_data(tokenizer, data, labels, max_length=500, multi_label=False):

    data_out = []

    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )

        if multi_label:
            data_out.append((encoded_dict['input_ids'],
                             encoded_dict['attention_mask'],
                             np.sort(labels[i])))
        else:
            data_out.append((encoded_dict['input_ids'],
                             encoded_dict['attention_mask'],
                             labels[i]))

    return TransformersDataset(data_out, multi_label=multi_label)
