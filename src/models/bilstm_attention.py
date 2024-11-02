# src/models/bilstm_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BiLSTMAttentionModel(nn.Module):
    def __init__(
        self, 
        embedding_matrix: np.ndarray, 
        hidden_size: int, 
        output_size: int, 
        num_layers: int = 1, 
        freeze_embeddings: bool = False
    ) -> None:
        super(BiLSTMAttentionModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=freeze_embeddings
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        outputs, _ = self.lstm(embedded)  # Shape: [batch_size, seq_len, hidden_size*2]
        # Compute attention weights
        attn_weights = F.softmax(
            self.attention(outputs).squeeze(2), dim=1
        )  # Shape: [batch_size, seq_len]
        # Compute weighted sum of outputs
        attn_output = torch.bmm(
            outputs.transpose(1, 2), attn_weights.unsqueeze(2)
        ).squeeze(2)  # Shape: [batch_size, hidden_size*2]
        out = self.fc(attn_output)
        return out
