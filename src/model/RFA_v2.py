import torch.nn as nn
import torch

class RFA_MLP(nn.Module):
    def __init__(self, num_feats, num_pdfs):
        super().__init__()
        self.fc1 = nn.Linear(num_feats, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, num_pdfs)
    
    def forward(self, x):
        a1 = self.fc1(x)
        h1 = torch.relu(a1)
        
        a2 = self.fc2(h1)
        h2 = torch.relu(a2)
        
        a3 = self.fc3(h2)
        h3 = torch.relu(a3)
        
        logits = self.fc4(h3)
        
        return a1, h1, a2, h2, a3, h3, logits