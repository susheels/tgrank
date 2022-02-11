import torch

class MLP(torch.nn.Module):
	def __init__(self, in_dim, h_dim=128, drop=0.0):
		super().__init__()
		self.fc_1 = torch.nn.Linear(in_dim, h_dim)
		self.fc_2 = torch.nn.Linear(h_dim, 1)
		self.act = torch.nn.ReLU()
		self.dropout = torch.nn.Dropout(p=drop, inplace=False)
		self.sigmoid = torch.nn.Sigmoid()

		torch.nn.init.xavier_normal_(self.fc_1.weight)
		torch.nn.init.xavier_normal_(self.fc_2.weight)

	def forward(self, x):
		x = self.act(self.fc_1(x))
		x = self.dropout(x)
		x = self.fc_2(x)
		return self.sigmoid(x)
