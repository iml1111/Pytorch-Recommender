from copy import deepcopy
import torch
from tqdm import trange, tqdm
import numpy as np
from data_loader import BatchIterator


class Trainer:
    """모델 학습기 클래스"""

    def __init__(
        self,
        model,
        optim,
        crit,
        device
    ):
        self.model = model
        self.optim = optim
        self.crit = crit
        # batch data가 저장될 device
        # TODO: 모델의 소속 디바이스를 검출할 수 없을까?
        self.device = device 

    def _train(self, data_loader, config):
        """for 1 Epoch Train"""
        self.model.train()
        total_loss = 0
        batches = BatchIterator(
            data_loader.train_x, 
            data_loader.train_y,
            config.batch_size,
            self.device
        )
        progress = trange(batches.iteration, desc="train-loss:nan | param:nan | g_param:nan")
        for _ in progress:
            x, y = next(batches)
            y_hat = self.model(
                x[:, 0], x[:, 1],
                data_loader.min_rate,
                data_loader.max_rate,
            )
            loss = self.crit(y_hat, y)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Train Process Printing...
            # 현재 loss, 현재 이터레이션, 각종 norm 보고

            progress.set_description(
                "train-loss:%.4e | param:%.4e | g_param:%.4e" % (
                    float(loss),
                    float(self.get_parameter_norm(self.model.parameters())),
                    float(self.get_grad_norm(self.model.parameters())),
                ) 
            )
            progress.refresh()

            total_loss += float(loss)

        return total_loss / batches.iteration


    def _valid(self, data_loader, config):
        """for 1 Epoch Validation"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            batches = BatchIterator(
                data_loader.valid_x,
                data_loader.valid_y,
                config.batch_size,
                self.device
            )
            progress = trange(batches.iteration, desc="valid-loss: nan")
            for _ in progress:
                x, y = next(batches)
                y_hat = self.model(
                    x[:, 0], x[:, 1],
                    data_loader.min_rate,
                    data_loader.max_rate,
                )
                loss = self.crit(y_hat, y)

                # Train Process Printing...
                # TODO: 현재 loss, 현재 이터레이션, 각종 norm 보고
                progress.set_description(
                    "valid-loss: %.4e" % float(loss)
                )
                progress.refresh()

                total_loss += float(loss)

        return total_loss / batches.iteration

    def train(self, data_loader, config):
        """Train"""
        lowest_loss =np.inf
        best_model = None
        
        for epoch in range(config.n_epochs):
            train_loss = self._train(data_loader, config)
            valid_loss = self._valid(data_loader, config)

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e valid_loss=%.4e lowest_loss=%.4e" % (
                epoch + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        self.model.load_state_dict(best_model)

    @staticmethod
    def get_grad_norm(parameters, norm_type=2):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = 0

        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        return total_norm

    @staticmethod
    def get_parameter_norm(parameters, norm_type=2):
        total_norm = 0

        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        return total_norm






