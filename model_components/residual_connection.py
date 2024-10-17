import torch
import torch.nn as nn
from add_and_norm import AddAndNorm


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = AddAndNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))