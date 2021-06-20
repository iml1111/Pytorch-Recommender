from copy import deepcopy
import torch
from data_loader import KMRDDataLoader, BatchIterator


class Trainer:
    """모델 학습기 클래스"""

    def __init__(
        self,
        model,
        optim,
        crit,
        data_loader: KMRDDataLoader,
        device
    ):
        self.model = model
        self.optim = optim
        self.crit = crit
        self.data_loader = data_loader
        self.device = device

    def _train(self, x, y, config):
        """for 1 Epoch Train"""
        self.model.train()
        total_loss = 0
        batches = BatchIterator(
            data_loader.train_x,
            data_loader.train_y,
            config.batch_size,
            self.device
        )
        for x, y in batches:
            y_hat = self.model(x)
            loss = self.crit(y_hat, y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Train Process Printing...
            # 현재 loss, 현재 이터레이션, 각종 norm 보고

            total_loss += float(loss)

        return total_loss / len()


    def _valid(self, x, y, config):
        """for 1 Epoch Validation"""
        self.model.eval()

    def train(self, train_data, valid_data, config):
        pass