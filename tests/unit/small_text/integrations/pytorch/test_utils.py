import unittest

import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from small_text.integrations.pytorch.models.kimcnn import KimCNN
    from small_text.integrations.pytorch.utils.misc import _assert_layer_exists
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
class PytorchIntegrationUtilsTest(unittest.TestCase):

    def test_assert_layer_exists(self):
        model = KimCNN(10_000, 20)
        _assert_layer_exists(model, 'embedding')

    def test_assert_layer_exists_fail(self):
        model = KimCNN(10_000, 20)
        with self.assertRaises(ValueError):
            _assert_layer_exists(model, 'other.layer')
