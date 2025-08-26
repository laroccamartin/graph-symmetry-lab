from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchMPN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.W_msg = nn.ParameterList([nn.Parameter(torch.randn(in_dim if i==0 else hidden_dim, hidden_dim)*0.1) for i in range(layers)])
        self.W_self = nn.ParameterList([nn.Parameter(torch.randn(in_dim if i==0 else hidden_dim, hidden_dim)*0.1) for i in range(layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(layers)])
        self.W_out = nn.Parameter(torch.randn(hidden_dim, 1)*0.1)
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, A: torch.Tensor, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        A: (B, N, N), X: (B, N, D), mask: (B, N)
        Returns y: (B, 1)
        """
        H = X
        for i in range(self.layers):
            AX = torch.bmm(A, H)               # (B, N, D or H)
            H = AX @ self.W_msg[i] + H @ self.W_self[i] + self.b[i]
            H = F.relu(H)
            H = H * mask.unsqueeze(-1)
        # masked mean pooling
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
        y = pooled @ self.W_out + self.b_out   # (B, 1)
        return y
