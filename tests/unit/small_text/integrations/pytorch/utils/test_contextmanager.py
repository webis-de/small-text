import unittest
import pytest

from unittest.mock import patch

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.utils.contextmanager import inference_mode
    from small_text.integrations.pytorch.utils import contextmanager
except (PytorchNotFoundError, ModuleNotFoundError):
    pass


@pytest.mark.pytorch
class ContextmanagerTest(unittest.TestCase):

    def test_inference_mode(self):
        with patch('torch.__version__', new='1.9.0'), \
            patch('torch.inference_mode', wraps=torch.inference_mode) as inference_mode_spy, \
            patch('torch.no_grad', wraps=torch.no_grad) as no_grad_spy:

            with inference_mode():
                pass

            inference_mode_spy.assert_called()
            no_grad_spy.assert_not_called()

    def test_inference_mode_with_fallback(self):
        with patch('torch.__version__', new='1.8.0'), \
            patch('torch.inference_mode', wraps=torch.inference_mode) as inference_mode_spy, \
            patch('torch.no_grad', wraps=torch.no_grad) as no_grad_spy:

            with inference_mode():
                pass

            no_grad_spy.assert_called()
            inference_mode_spy.assert_not_called()
