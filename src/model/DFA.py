import torch
import torch.nn as nn


class DFA_MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        a1 = self.fc1(x)
        h1 = torch.relu(a1)
        a2 = self.fc2(h1)
        h2 = torch.relu(a2)
        logits = self.fc3(h2)
        return a1, h1, a2, h2, logits
