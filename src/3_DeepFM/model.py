import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FeaturesLinear(nn.Module):

    def __init__(self, user_movie_cnts, output_dim=1):
        super().__init__()
        # 하나의 임베딩 레이어에서 유저의 임베딩벡터와 아이템의 임베딩벡터를 공유함
        self.fc = nn.Embedding(sum(user_movie_cnts), output_dim) # (user_cnt + moive_cnt) -> 1
        self.bias = nn.Parameter(torch.zeros((output_dim,))) # (1,) 마지막 값 결과에 더함
        # [0, user_cnt]: User 데이터의 임베딩 인덱스 범위 
        self.offsets = np.array((0, *np.cumsum(user_movie_cnts)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        x = (batch_size, 2) [user, movie]
        """
        
        """
        new_tensor = (1, 2) [0, 450]
        x = (batch_size, 2)
        new_tensor(offsets)란?
        - item의 임베딩 인덱스를 뒤로 밀어주기 위해 user_cnt만큼 값을 밀어줌
        - user는 0부터 시작하므로 그대로 놔둠
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # x = (batch_size, 2)
        x = self.fc(x)
        # y = (batch_size, 1)
        y = torch.sum(x, dim=1) + self.bias
        return y


class FeaturesEmbedding(nn.Module):

    def __init__(self, user_movie_cnts, embed_dim):
        super().__init__()
        # 하나의 임베딩 레이어에서 유저의 임베딩벡터와 아이템의 임베딩벡터를 공유함
        self.emb = nn.Embedding(sum(user_movie_cnts), embed_dim) # (user_cnt + movie_cnt) -> dim
        self.offsets = np.array((0, *np.cumsum(user_movie_cnts)[:-1]), dtype=np.long) # [0, user_cnt]
        nn.init.xavier_uniform_(self.emb.weight.data) # 웨이트 초기화 알고리즘의 일종

    def forward(self, x):
        """
        x = (batch_size, 2) [user, movie]
        """
        # x = (batch_size, 2) 
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # y = (batch_size, 2, embed_dim) ?????? 직접 돌려봐야 할듯
        y = self.emb(x)
        return y


class FactorizationMachine(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        FM Layer -> 프로덕션만 수행할 뿐, 웨이트 학습을 하진 않음
        임베딩을 거쳐온 데이터를 그대로 넣어줌
        x = (batch_size, 2, embed_dim)
        """
        # 합의 제곱 (u+i)^2 = (batch_size, embed_dim)
        square_of_sum = torch.sum(x, dim=1) ** 2
        # 제곱의 합 u^2 + i^2 = (batch_size, embed_dim)
        sum_of_square = torch.sum(x ** 2, dim=1)
        # ix = (u+i)^2 - (u^2 + i^2) = (batch_size, embed_dim)
        ix = square_of_sum - sum_of_square
        # ix = (batch_size, 1)
        ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(embed_dim))
            input_dim = embed_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x = (batch_size, embed_dim)
        """
        # (batch_size, 1)
        return self.mlp(x)


class DeepFactorizationMachine(nn.Module):

    def __init__(self, user_movie_cnts, embed_dim, mlp_dims):
        super().__init__()
        self.linear = FeaturesLinear(user_movie_cnts)
        self.fm = FactorizationMachine()
        self.emb = FeaturesEmbedding(user_movie_cnts, embed_dim)
        self.embed_output_dim = len(user_movie_cnts) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims)

    def forward(self, x):
        """
        x = (batch_size, 2)
        """
        # embed_x = (batch_size, 2, embed_dim)
        embed_x = self.emb(x)
        # y = (batch_size, 1)
        y = (
            self.linear(x)
            + self.fm(embed_x)
            + self.mlp(embed_x.view(-1, self.embed_output_dim))
        )
        # (batch_size,)
        return torch.sigmoid(y)








