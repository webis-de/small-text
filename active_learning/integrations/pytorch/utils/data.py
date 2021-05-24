from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError


try:
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


def dataloader(data_set, batch_size, collate_fn, train=True):
    """
    Convenience method to obtain a `DataLoader`.

    Parameters
    ----------
    data_set : DataLoader
        The target dataset.
    batch_size : int
        Batch size.
    collate_fn : func
        The `collate-fn` required by `DataLoader`.
    train : bool
        Indicates if the dataloader is used for training or testing. For training random sampling
        is used, otherwise sequential sampling.

    Returns
    -------
    iter : DataLoader
        A DataLoader for the given `data_set`.
    """

    if train:
        base_sampler = RandomSampler(data_set)
    else:
        base_sampler = SequentialSampler(data_set)

    sampler = BatchSampler(
        base_sampler,
        batch_size=batch_size,
        drop_last=False)

    return DataLoader(data_set,
                      batch_size=None,
                      collate_fn=collate_fn,
                      sampler=sampler)
