import unittest

import pytest

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from active_learning.integrations.pytorch.models.kimcnn import KimCNN
    from active_learning.integrations.pytorch.utils.misc import assert_layer_exists
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
class PytorchIntegrationUtilsTest(unittest.TestCase):

    def test_assert_layer_exists(self):
        model = KimCNN(10_000, 20)
        assert_layer_exists(model, 'embedding')

    def test_assert_layer_exists_fail(self):
        model = KimCNN(10_000, 20)
        with self.assertRaises(ValueError):
            assert_layer_exists(model, 'other.layer')
