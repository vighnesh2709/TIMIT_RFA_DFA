import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_feats, num_pdfs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_feats, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_pdfs)
        )

    def forward(self, x):
        return self.net(x)
