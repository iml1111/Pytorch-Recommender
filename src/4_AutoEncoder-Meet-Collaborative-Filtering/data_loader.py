import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class KMRDDataset(Dataset):

	def __init__(self, data_path):

		data = pd.read_csv(data_path)
		self.user2idx = {origin: idx for idx, origin in enumerate(data.user.unique())}
		self.movie2idx = {origin: idx for idx, origin in enumerate(data.movie.unique())}

		self.min_rate = min(data.rate)
		self.max_rate = max(data.rate)

		self.user = [self.user2idx[u] for u in data.user.values]
		self.movie = [self.movie2idx[m] for m in data.movie.values]
		self.rating = data.rate.values

		input_tensor = torch.LongTensor([self.movie, self.user])
		self.data = torch.sparse.FloatTensor(
			input_tensor, torch.FloatTensor(self.rating),
			torch.Size([len(self.movie2idx), len(self.user2idx)]),
		).to_dense()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""AutoEncoder이므로, x만 있으면 됨"""
		return self.data[idx]