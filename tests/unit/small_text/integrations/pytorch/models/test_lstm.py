import unittest

import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.models.lstm import LSTM
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class LSTMTest(unittest.TestCase):

    def test_forward_returns_logits_with_expected_shape(self):
        model = LSTM(20, 5, num_classes=3, hidden_size=7, dropout=0.0)
        inputs = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])

        logits = model(inputs)

        self.assertEqual((2, 3), logits.shape)
        self.assertEqual(7, model.fc.in_features)

    def test_uses_supplied_embedding_matrix(self):
        embedding_matrix = torch.arange(60, dtype=torch.float32).reshape(12, 5)

        model = LSTM(12, 4, embed_dim=100, embedding_matrix=embedding_matrix)

        self.assertEqual((12, 5), model.embedding.weight.shape)
        self.assertTrue(torch.equal(embedding_matrix, model.embedding.weight))
        self.assertEqual(5, model.lstm.input_size)

    def test_embedding_layer_can_be_trainable_or_frozen(self):
        embedding_matrix = torch.rand(12, 5)

        trainable_model = LSTM(12, 4, embedding_matrix=embedding_matrix)
        frozen_model = LSTM(12, 4, embedding_matrix=embedding_matrix,
                            freeze_embedding_layer=True)

        self.assertTrue(trainable_model.embedding.weight.requires_grad)
        self.assertFalse(frozen_model.embedding.weight.requires_grad)

    def test_bidirectional_lstm_concatenates_final_hidden_states(self):
        hidden_size = 7
        model = LSTM(20, 5, hidden_size=hidden_size, num_classes=3,
                     bidirectional=True, dropout=0.0)
        inputs = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])

        pooled = model._forward_pooled(inputs)
        logits = model(inputs)

        self.assertEqual((2, hidden_size * 2), pooled.shape)
        self.assertEqual(hidden_size * 2, model.fc.in_features)
        self.assertEqual((2, 3), logits.shape)

    def test_pooled_representation_ignores_padding(self):
        for bidirectional in [False, True]:
            with self.subTest(bidirectional=bidirectional):
                model = LSTM(20, 5, hidden_size=4, bidirectional=bidirectional, dropout=0.0)
                model.eval()
                inputs = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 5, 6, 0]])

                pooled = model._forward_pooled(inputs)
                embedded = model.embedding(inputs[0, :2]).unsqueeze(0)
                sequence_outputs, (expected_hidden, cell_states) = model.lstm(embedded)
                if bidirectional:
                    expected = torch.cat((expected_hidden[-2], expected_hidden[-1]), dim=1)
                else:
                    expected = expected_hidden[-1]

                self.assertTrue(torch.allclose(expected, pooled[0]))
                self.assertEqual((2, 8 if bidirectional else 4), pooled.shape)
