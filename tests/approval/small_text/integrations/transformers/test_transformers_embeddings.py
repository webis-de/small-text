import unittest
import pytest

import numpy as np

from approvaltests import verify

from small_text import (
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
)
from small_text.integrations.pytorch.classifiers.base import AMPArguments
from tests.utils.datasets import twenty_news_transformers
from tests.utils.misc import random_seed


@pytest.mark.pytorch
class TransformerEmbeddingsApprovalTest(unittest.TestCase):

    @random_seed(seed=42, set_torch_seed=True)
    def test_embeddings(self, num_classes=3, device='cuda'):
        classifier_kwargs = {
            'num_epochs': 1,
            'device': device,
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            num_classes,
            classification_kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        with np.printoptions(precision=4, edgeitems=10, linewidth=np.inf):
            output = f'{clf.embed(train_set)}'
        verify(output)

    @random_seed(seed=42, set_torch_seed=True)
    def test_embeddings_amp(self, num_classes=3, device='cuda'):
        amp_args = AMPArguments(use_amp=True, device_type='cuda')
        classifier_kwargs = {
            'num_epochs': 1,
            'device': device,
            'amp_args': amp_args
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            num_classes,
            classification_kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        with np.printoptions(precision=4, edgeitems=10, linewidth=np.inf):
            output = f'{clf.embed(train_set)}'
        verify(output)
