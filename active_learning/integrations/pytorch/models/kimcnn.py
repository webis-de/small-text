"""KimCNN implementation using pytorch.

This implementation is based on:
https://github.com/Johnxjp/cnn_text_classification/tree/d05e8ede5bbfd2a4de3c2df92ea705cab0e803f2
by John Lingi (Johnxjp)
(MIT-licensed)
"""
from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise PytorchNotFoundError('Could not import pytorch or one of its dependencies: ' + str(e))


class KimCNN(nn.Module):
    """

    Parameters
    ----------
    vocabulary_size : int

    max_seq_length : int

    num_classes : int
        Number of output classes.

    embedding_matrix : 2D FloatTensor

    """
    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100,
                 embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5,
                 embedding_matrix=None, freeze_embedding_layer=False):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = 1
        self.n_kernels = len(kernel_heights)
        self.pool_sizes = [(max_seq_length - K, 1) for K in kernel_heights]
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

        # Assumes vocab size is same as embedding matrix size. Therefore should
        # contain special tokens e.g. <pad>
        self.embedding = nn.Embedding(
            vocabulary_size, embed_dim, padding_idx=padding_idx
        )

        if embedding_matrix is not None:
            # Load pre-trained weights. Should be torch FloatTensor
            self.embedding = self.embedding.from_pretrained(embedding_matrix.float(),
                                                            padding_idx=padding_idx)

        self.embedding.weight.requires_grad = not freeze_embedding_layer

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=(K, embed_dim)
                )
                for K in kernel_heights
            ]
        )
        self.pools = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=pool_size)
                for pool_size in self.pool_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.out_channels * self.n_kernels, self.num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.LongTensor or torch.cuda.LongTensor
            input tensor (batch_size, max_sequence_length) with padded sequences of word ids
        """
        batch_size = x.size(0)
        assert x.size(1) == self.max_seq_length

        x = self.embedding(x)
        x = x.unsqueeze(dim=1)

        out_tensors = []
        for conv, pool in zip(self.convs, self.pools):
            activation = pool(F.relu(conv(x)))
            out_tensors.append(activation)

        # (batch_size, out * n_kernels, 1, 1)
        x = torch.cat(out_tensors, dim=1)

        x = x.view(batch_size, -1)
        x = self.dropout(x)

        return self.fc(x)
