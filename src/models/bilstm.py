from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

class BiLSTMModel(nn.Module):
    def __init__(
        self, 
        embedding_matrix: np.ndarray, 
        hidden_size: int, 
        output_size: int, 
        num_layers: int = 1
    ) -> None:
        super(BiLSTMModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix)
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        # Concatenate the final forward and backward hidden states
        hidden_fw = hidden[-2, :, :]
        hidden_bw = hidden[-1, :, :]
        hidden_cat = torch.cat((hidden_fw, hidden_bw), dim=1)
        out = self.fc(hidden_cat)
        return out
