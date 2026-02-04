import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,nums_feats,num_pdfs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nums_feats, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_pdfs)
        )
    def forward(self,x):
        return self.net(x)