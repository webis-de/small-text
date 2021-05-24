from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError as _PytorchNotFoundError

try:
    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
except _PytorchNotFoundError:
    pass

try:
    import transformers
    from active_learning.integrations.transformers.datasets import TransformersDataset
except (_PytorchNotFoundError, ModuleNotFoundError):
    pass
