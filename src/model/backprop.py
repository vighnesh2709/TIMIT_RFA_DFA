import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_feats: int, num_pdfs: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_feats, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pdfs),
        )

    def forward(self, x):
        return self.net(x)
