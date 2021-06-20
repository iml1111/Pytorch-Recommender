import pandas as pd
import numpy as np
import torch
from torch import LongTensor, FloatTensor
from sklearn.model_selection import train_test_split


class KMRDDataLoader:
    """
    Load KMRD Dataset
    """

    def __init__(self, data_path, valid_raito=0.2):
        train_df, valid_temp_df = self._read_dataset(data_path, valid_raito)

        # Min, Max Rating
        self.min_rate = min(train_df.rate)
        self.max_rate = train_df.rate.max()

        # Refine User Info
        self.users = train_df.user.unique()
        self.num_users = len(self.users)
        self.user2idx = {origin: idx for idx, origin in enumerate(self.users)}

        # Refine Item Info
        self.items = train_df.movie.unique()
        self.num_items = len(self.items)
        self.item2idx = {origin: idx for idx, origin in enumerate(self.items)}

        valid_df = valid_temp_df[
            valid_temp_df.user.isin(self.users)
            & valid_temp_df.movie.isin(self.items)
        ]
        # Make Train set
        self.train_x, self.train_y = self._cast_tensor(train_df)
        self.valid_x, self.valid_y = self._cast_tensor(valid_df)

    @staticmethod
    def _read_dataset(data_path, valid_ratio):
        return train_test_split(
            pd.read_csv(data_path),
            test_size=valid_ratio,
            random_state=1111, # random fix
            shuffle=True
        )

    def _cast_tensor(self, data):
        return (
            LongTensor(
                pd.DataFrame({
                    'user': data.user.map(self.user2idx),
                    'item': data.movie.map(self.item2idx)
                }).values
            ),
            FloatTensor(
                data['rate'].astype(np.float32).values
            ).view(-1, 1)
        )


class BatchIterator:
    """
    해당 데이터셋으로 배치 사이즈만큼 나누어서 iteration 해줌.
    """

    def __init__(self, x, y, batch_size, device):
        
        # Data Shuffle & Slicing
        indices = torch.randperm(x.size(0))
        x = torch.index_select(x, dim=0, index=indices)
        y = torch.index_select(y, dim=0, index=indices)
        
        self.x = x.split(batch_size, dim=0)
        self.y = y.split(batch_size, dim=0)
        self.device = device
        self.batch_size = batch_size
        self.iteration = len(self.x)
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cur >= self.iteration:
            raise StopIteration()
        self.cur += 1
        return (
            self.x[self.cur - 1].to(self.device), 
            self.y[self.cur - 1].to(self.device)
        )

