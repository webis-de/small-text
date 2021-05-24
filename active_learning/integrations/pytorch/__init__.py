from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
except PytorchNotFoundError:
    pass
