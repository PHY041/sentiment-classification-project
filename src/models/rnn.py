from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

# class RNNModel(nn.Module):
#     def __init__(
#         self, 
#         embedding_matrix: np.ndarray, 
#         hidden_size: int, 
#         output_size: int
#     ) -> None:
#         super(RNNModel, self).__init__()
#         vocab_size, embedding_dim = embedding_matrix.shape
#         self.embedding = nn.Embedding.from_pretrained(
#             torch.FloatTensor(embedding_matrix), freeze=True
#         )
#         self.rnn = nn.RNN(
#             input_size=embedding_dim, 
#             hidden_size=hidden_size, 
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, output_size)

#     # def forward(self, x: torch.Tensor) -> torch.Tensor:
#     #     embedded = self.embedding(x)
#     #     output, hidden = self.rnn(embedded)
#     #     # Use the last hidden state
#     #     out = self.fc(hidden.squeeze(0))
#     #     return out

#     #modified forward function to use average pooling instead of the last hidden state
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         embedded = self.embedding(x)
#         output, hidden = self.rnn(embedded)
#         # Average pooling over time steps
#         out = self.fc(output.mean(dim=1))
#         return out

import torch
import torch.nn as nn
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size, freeze=False):
        super(RNNModel, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=freeze
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            nonlinearity='tanh',
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights for fc layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
        # Initialize weights for RNN layer
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        output, _ = self.rnn(embedded)  # Output shape: [batch_size, seq_len, hidden_size]
        # Use the last time step's output
        output = output[:, -1, :]  
        return self.fc(output)     

