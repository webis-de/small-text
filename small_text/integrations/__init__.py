from small_text.integrations.pytorch.exceptions import PytorchNotFoundError as _PytorchNotFoundError

try:
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
except _PytorchNotFoundError:
    pass

try:
    import transformers
    from small_text.integrations.transformers.datasets import TransformersDataset
except (_PytorchNotFoundError, ModuleNotFoundError):
    pass
