import torch.nn as nn
import torch

class RFA_MLP(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim):
		super().__init__()
		self.fc1 = nn.Linear(in_dim, hidden_dim, bias = True)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias = True)
		self.fc3 = nn.Linear(hidden_dim, out_dim, bias = True)

	def forward(self,x):
		a1 = self.fc1(x)
		h1 = torch.relu(a1)

		a2 = self.fc2(h1)
		h2 = torch.relu(a2)

		a3 = self.fc3(h2)

		return a1,h1,a2,h2,a3
