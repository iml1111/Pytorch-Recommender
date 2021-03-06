from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class KMRDDataset(Dataset):

	def __init__(self, data_path):

		data = pd.read_csv(data_path)

		user2idx = {origin: idx for idx, origin in enumerate(data.user.unique())}
		movie2idx = {origin: idx for idx, origin in enumerate(data.movie.unique())}
		data['user'] = data['user'].apply(lambda x: user2idx[x])
		data['moive'] = data['movie'].apply(lambda x: movie2idx[x])
		data = data.to_numpy()[:, :3] # (data_num, 3)

		self.x = data[:, :2].astype(np.int)
		self.y = self._preprocess_target(data[:, 2:]).astype(np.float32)
		self.user_movie_cnts = np.max(self.x, axis=0) + 1
		
		self.user_field_idx = np.array((0,), dtype=np.long)
		self.movie_field_idx = np.array((1,), dtype=np.long)

	def __len__(self):
		return self.y.shape[0]

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def _preprocess_target(self, target):
		target[target <= 7] = 0
		target[target > 7] = 1
		return target
