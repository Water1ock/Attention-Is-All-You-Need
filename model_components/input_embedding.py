import math
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        normalized_embeddings = embeddings * math.sqrt(self.d_model)
        return normalized_embeddings
