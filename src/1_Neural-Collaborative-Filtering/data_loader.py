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
        self.min_rating = min(train_df.rate)
        self.max_rating = train_df.rate.max()

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
        self.device =device
        self.batch_size = batch_size
        self.iteration = x.shape[0] // batch_size
        self.x, self.y = x, y
        self.cur = 0
        # TODO: 기존 방식으로 체인지하기 (제너레이팅 X)
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get_batch(self, data):
        """현재 Batch만 Device로 올림"""
        return data[
            (self.cur - 1) * self.batch_size:
            self.cur * self.batch_size
        ].to(self.device)

    def next(self):
        if self.cur >= self.iteration:
            raise StopIteration()
        self.cur += 1
        # 배치 데이터 슬라이싱 및 셔플
        x, y = self.get_batch(self.x), self.get_batch(self.y)
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices)
        y = torch.index_select(y, dim=0, index=indices)
        return x, y

"""
def _train(self, x, y, config):
        '''
        한 에포크당 학습 과정
        '''

        # 학습 모드 On
        # 이걸 해야 학습이 모델에 반영됨
        self.model.train()

        # 학습 데이터 셔플
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices)
        y = torch.index_select(y, dim=0, index=indices)

        # 배치 사이즈에 맞춰서 자르기
        x = x.split(config.batch_size, dim=0)
        y = y.split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            pred_y_i = self.model(x_i)
            # squeeze : (N, 1) -> (N,)로 변환
            # print("-----")
            # print(pred_y_i.size())
            # print(y_i.squeeze().size())
            # print("-----")
            loss = self.crit(pred_y_i, y_i.squeeze())

            # 그래디언트 초기화 및 역전파 진행
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (
                    i + 1,
                    len(x),
                    float(loss)
                ))
            # 효율적인 메모리 연산을 위해 float로 캐스팅
            total_loss += float(loss)

        return total_loss / len(x)
"""