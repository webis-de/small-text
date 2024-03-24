from unittest import mock


def autocast_asserting_decorator(method, expected_autocast_state, test_case):
    import torch
    method_mock = mock.MagicMock()
    def wrapper(self, *args, **kwargs):
        test_case.assertEqual(expected_autocast_state, torch.is_autocast_enabled())
        method_mock(*args, **kwargs)
        return method(self, *args, **kwargs)
    return wrapper
