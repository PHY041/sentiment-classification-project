from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNModel(nn.Module):
    def __init__(
        self, 
        embedding_matrix: np.ndarray, 
        output_size: int, 
        kernel_sizes: List[int] = [3, 4, 5],
        num_filters: int = 100, 
        dropout: float = 0.5
    ) -> None:
        super(CNNModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix)
        )
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=num_filters, 
                kernel_size=(k, embedding_dim)
            )
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # [batch_size, seq_len, emb_dim]
        embedded = embedded.unsqueeze(1)  # Add channel dimension: [batch_size, 1, seq_len, emb_dim]
        conved = [
            F.relu(conv(embedded)).squeeze(3)  # [batch_size, num_filters, seq_len - k + 1]
            for conv in self.convs
        ]
        # Max-over-time pooling
        pooled = [
            F.max_pool1d(c, c.size(2)).squeeze(2)  # [batch_size, num_filters]
            for c in conved
        ]
        cat = torch.cat(pooled, dim=1)  # [batch_size, num_filters * len(kernel_sizes)]
        out = self.dropout(cat)
        out = self.fc(out)
        return out
