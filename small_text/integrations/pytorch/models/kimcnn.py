#
# MIT License
#
# Copyright (c) 2019 John Lingi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
KimCNN implementation is based on:
https://github.com/Johnxjp/cnn_text_classification/tree/d05e8ede5bbfd2a4de3c2df92ea705cab0e803f2
by John Lingi (@Johnxjp, MIT-licensed)
"""
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
except ImportError as e:
    raise PytorchNotFoundError('Could not import pytorch or one of its dependencies: ' + str(e))


class KimCNN(nn.Module):

    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, out_channels=100,
                 embed_dim=300, padding_idx=0, kernel_heights=[3, 4, 5], dropout=0.5,
                 embedding_matrix=None, freeze_embedding_layer=False):
        """
        Parameters
        ----------
        vocabulary_size : int
            Number of tokens contained in of the vocabulary.
        max_seq_length : int
            Maximum sequence length.
        num_classes : int, default=2
            Number of output classes.
        out_channels : int, default=100
            Number of output channels.
        embed_dim : int, default=300
            Number of dimensions of a single embedding.
        padding_idx : int, default=0
            Padding index (passed to the embedding layer).
        kernel_heights : list of int, default=[3, 4, 5]
            Kernels heights for the convolutions.
        dropout : float, default=0.5
            Dropout Probability.
        embedding_matrix : torch.FloatTensor or torch.cuda.FloatTensor, default=None
            Embedding matrix in the shape (vocabulary_size, embed_dim).
        freeze_embedding_layer : bool, default=False
            Training adapts the embedding matrix if `True`, otherwise the embeddings are frozen.
        """
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = 1
        self.num_kernels = len(kernel_heights)
        self.pool_sizes = [(max_seq_length - k, 1) for k in kernel_heights]
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
                    kernel_size=(k, embed_dim)
                )
                for k in kernel_heights
            ]
        )
        self.pools = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=pool_size)
                for pool_size in self.pool_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.out_channels * self.num_kernels, self.num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.LongTensor or torch.cuda.LongTensor
            input tensor (batch_size, max_sequence_length) with padded sequences of word ids
        """
        x = self._forward_pooled(x)
        return self._dropout_and_fc(x)

    def _forward_pooled(self, x):
        assert x.size(1) == self.max_seq_length

        x = self.embedding(x)
        x = x.unsqueeze(dim=1)

        out_tensors = []
        for conv, pool in zip(self.convs, self.pools):
            activation = pool(F.relu(conv(x)))
            out_tensors.append(activation)

        x = torch.cat(out_tensors, dim=1)
        return x.view(x.size(0), -1)

    def _dropout_and_fc(self, x):
        x = self.dropout(x)
        return self.fc(x)
