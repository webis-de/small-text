import unittest
import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNN
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class KimCNNInitTest(unittest.TestCase):

    DEFAULT_KERNEL_HEIGHTS = [3, 4, 5]

    def test_init_parameters_default(self):

        vocab_size = 1000
        max_seq_length = 50

        model = KimCNN(vocab_size, max_seq_length)

        pool_sizes = [(47, 1), (46, 1), (45, 1)]

        # Parameters
        self.assertEqual(100, model.out_channels)
        self.assertEqual(1, model.in_channels)
        self.assertEqual(3, model.num_kernels)
        self.assertEqual(pool_sizes, model.pool_sizes)
        self.assertEqual(max_seq_length, model.max_seq_length)
        self.assertEqual(2, model.num_classes)

        # Modules
        self.assertTrue(model.embedding.weight.requires_grad)

        self.assertEqual(0, model.embedding.padding_idx)
        self.assertEqual(vocab_size, model.embedding.num_embeddings)
        self.assertEqual(300, model.embedding.embedding_dim)

        self.assertEqual(len(pool_sizes), len(model.pools))
        for i, pool in enumerate(model.pools):
            self.assertEqual(pool_sizes[i], pool.kernel_size)

        self.assertEqual(3, len(model.convs))
        for i, conv in enumerate(model.convs):
            self.assertEqual(1, conv.in_channels)
            self.assertEqual(100, conv.out_channels)
            self.assertEqual((self.DEFAULT_KERNEL_HEIGHTS[i], 300), conv.kernel_size)

        self.assertEqual(0.5, model.dropout.p)
        self.assertEqual(300, model.fc.in_features)
        self.assertEqual(2, model.fc.out_features)

    def test_init_parameters_specific(self):
        vocab_size = 1000
        max_seq_length = 50

        num_classes = 3
        out_channels = 200
        embed_dim = 150
        padding_idx = 1
        kernel_heights = [4, 5]
        fc_dropout = 0.1
        embedding_matrix = None
        freeze_embedding_layer = True

        pool_sizes = [(46, 1), (45, 1)]

        model = KimCNN(vocab_size, max_seq_length, num_classes=num_classes,
                       out_channels=out_channels, embed_dim=embed_dim, padding_idx=padding_idx,
                       kernel_heights=kernel_heights, dropout=fc_dropout,
                       embedding_matrix=embedding_matrix,
                       freeze_embedding_layer=freeze_embedding_layer)

        # Parameters
        self.assertEqual(out_channels, model.out_channels)
        self.assertEqual(1, model.in_channels)
        self.assertEqual(2, model.num_kernels)
        self.assertEqual(pool_sizes, model.pool_sizes)
        self.assertEqual(max_seq_length, model.max_seq_length)
        self.assertEqual(num_classes, model.num_classes)

        # Modules
        self.assertFalse(model.embedding.weight.requires_grad)

        self.assertEqual(padding_idx, model.embedding.padding_idx)
        self.assertEqual(vocab_size, model.embedding.num_embeddings)
        self.assertEqual(embed_dim, model.embedding.embedding_dim)

        self.assertEqual(2, len(model.convs))
        for i, conv in enumerate(model.convs):
            self.assertEqual(1, conv.in_channels)
            self.assertEqual(out_channels, conv.out_channels)
            self.assertEqual((kernel_heights[i], embed_dim), conv.kernel_size)

        self.assertEqual(len(pool_sizes), len(model.pools))
        for i, pool in enumerate(model.pools):
            self.assertEqual(pool_sizes[i], pool.kernel_size)

        self.assertEqual(fc_dropout, model.dropout.p)
        self.assertEqual(400, model.fc.in_features)
        self.assertEqual(num_classes, model.fc.out_features)

    def test_init_with_embedding(self):

        vocab_size = 1000
        max_seq_length = 50

        fake_embedding = torch.rand(1000, 100, device='cpu')

        pool_sizes = [(47, 1), (46, 1), (45, 1)]

        model = KimCNN(vocab_size, max_seq_length, embedding_matrix=fake_embedding)

        # Parameters
        self.assertEqual(100, model.out_channels)
        self.assertEqual(1, model.in_channels)
        self.assertEqual(3, model.num_kernels)
        self.assertEqual(pool_sizes, model.pool_sizes)
        self.assertEqual(max_seq_length, model.max_seq_length)
        self.assertEqual(2, model.num_classes)

        # Modules
        self.assertTrue(model.embedding.weight.requires_grad)

        self.assertEqual(0, model.embedding.padding_idx)
        self.assertEqual(fake_embedding.size(0), model.embedding.num_embeddings)
        self.assertEqual(fake_embedding.size(1), model.embedding.embedding_dim)

        self.assertEqual(3, len(model.convs))
        for i, conv in enumerate(model.convs):
            self.assertEqual(1, conv.in_channels)
            self.assertEqual(100, conv.out_channels)
            self.assertEqual((self.DEFAULT_KERNEL_HEIGHTS[i], 300), conv.kernel_size)

        self.assertEqual(len(pool_sizes), len(model.pools))
        for i, pool in enumerate(model.pools):
            self.assertEqual(pool_sizes[i], pool.kernel_size)

        self.assertEqual(0.5, model.dropout.p)
        self.assertEqual(300, model.fc.in_features)
        self.assertEqual(2, model.fc.out_features)
