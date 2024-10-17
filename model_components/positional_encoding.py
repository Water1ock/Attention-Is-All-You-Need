import torch
import torch.nn as nn
from constants import POSITIONAL_SCALE


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        positional_encoding_matrix = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        i_indices = torch.arange(0, d_model).float()
        div_term = torch.pow(POSITIONAL_SCALE, 2*i_indices/d_model)

        positional_encoding_matrix[:, 0::2] = torch.sin(position / div_term)
        positional_encoding_matrix[:, 1::2] = torch.cos(position / div_term)

        positional_encoding_matrix = positional_encoding_matrix.unsqueeze(0)
        self.register_buffer('positional_encoding_matrix', positional_encoding_matrix)

    def forward(self, embeddings):
        embeddings = embeddings + (self.positional_encoding_matrix[:, :embeddings.shape[1], :]).requires_grad(False)
        return self.dropout(embeddings)
