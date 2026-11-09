from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise PytorchNotFoundError('Could not import pytorch or one of its dependencies: ' + str(e))


class LSTM(nn.Module):

    def __init__(self, vocabulary_size, max_seq_length, num_classes=2, hidden_size=100,
                 embed_dim=300, padding_idx=0, dropout=0.5, embedding_matrix=None,
                 freeze_embedding_layer=False, bidirectional=False):

        super().__init__()

        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            vocabulary_size, embed_dim, padding_idx=padding_idx
        )
        if embedding_matrix is not None:
            self.embedding = self.embedding.from_pretrained(
                embedding_matrix.float(), padding_idx=padding_idx
            )
        self.embedding.weight.requires_grad = not freeze_embedding_layer

        self.lstm = nn.LSTM(
            self.embedding.embedding_dim, hidden_size, batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    @property
    def device(self):
        return list(self.embedding.parameters())[0].device

    def forward(self, x):
        x = self._forward_pooled(x)
        return self._dropout_and_fc(x)

    def _forward_pooled(self, x):
        assert x.size(1) == self.max_seq_length

        lengths = x.ne(self.embedding.padding_idx).sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError('Cannot obtain an LSTM representation for an empty sequence.')

        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sequence_outputs, (hidden, cell_states) = self.lstm(x)

        if self.bidirectional:
            return torch.cat((hidden[-2], hidden[-1]), dim=1)
        return hidden[-1]

    def _dropout_and_fc(self, x):
        x = self.dropout(x)
        return self.fc(x)
