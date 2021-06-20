"""
# Neural Collaborative Filtering
Paper: https://arxiv.org/pdf/1708.05031.pdf
Original Code: https://github.com/hexiangnan/neural_collaborative_filtering

논문은 0과 1로 user-item interaction으로 matrix을 나타내고 학습했으나,
본 코드에서는 rating을 직접 예측함.
"""

import argparse
from pprint import pprint
import torch
from torch import optim
import torch.nn as nn
from model import NeuralMatrixFactorization
from data_loader import KMRDDataLoader
from trainer import Trainer

KMRD_SMALL_DATA_PATH = "../data/kmrd/kmr_dataset/datafile/kmrd-small/rates.csv"


def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        default='./models/model.pth',
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--data_path',
        default=KMRD_SMALL_DATA_PATH,
        help='Dataset Path, Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=200,
        help='Embedding Latent Vector Size. Default=%(default)s'
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=30,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    config = p.parse_args()
    return config


def main(config):
    print("# Config")
    pprint(vars(config))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_loader = KMRDDataLoader(config.data_path)

    print("Train:", data_loader.train_x.shape[0])
    print("Valid:", data_loader.valid_x.shape[0])

    model = NeuralMatrixFactorization(
        data_loader.num_users,
        data_loader.num_items,
        config.hidden_size
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.MSELoss().to(device)

    trainer = Trainer(model, optimizer, crit, device)
    trainer.train(data_loader, config)

    torch.save(
        {
            'model': trainer.model.state_dict(),
            'config':config
        },
        config.model_fn
    )


if __name__ == '__main__':
    config = define_argparser()
    main(config)
