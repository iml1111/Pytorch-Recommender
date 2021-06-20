import torch
import torch.nn as nn


class NeuralMatrixFactorization(nn.Module):

	def __init__(self, n_users, n_items, hidden_size):
		super().__init__()
		self.user_emb = nn.Embedding(n_users, hidden_size)
		self.item_emb = nn.Embedding(n_items, hidden_size)
		self.layers = nn.Sequential(
			nn.Linear(hidden_size * 2, 128),
			nn.LeakyReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
			nn.LeakyReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
			nn.LeakyReLU(),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
			nn.LeakyReLU(),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
			nn.LeakyReLU(),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
			nn.Sigmoid()
		)

	def forward(self, users, items, min_rate=0.5, max_rate=5.0):
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



