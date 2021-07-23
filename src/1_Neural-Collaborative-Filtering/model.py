import torch
import torch.nn as nn

# https://wikidocs.net/61374
class NeuralMatrixFactorization(nn.Module):

	def __init__(self, n_users, n_items, hidden_size):
		super().__init__()
		self.user_emb = nn.Embedding(n_users, hidden_size)
		self.item_emb = nn.Embedding(n_items, hidden_size)
		self.layers = nn.Sequential(
			nn.Linear(hidden_size * 2, 64),
			nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(64, 32),
			nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(32, 16),
			nn.ReLU(),
            nn.Dropout(.3),
            
            nn.ReLU(),
            nn.Linear(16, 1),
			nn.Sigmoid()
		)

	def forward(self, users, items, min_rate=1.0, max_rate=10.0):
		"""
		users = (bs, 1)
		items = (bs, 1)
		"""
		# x = (bs, hs * 2)
		x = torch.cat([
			self.user_emb(users), 
			self.item_emb(items),
		], dim=1)
		y = self.layers(x)
		return (y * (max_rate - min_rate)) + min_rate



