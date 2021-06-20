"""
# Neural Collaborative Filtering
Paper: https://arxiv.org/pdf/1708.05031.pdf
Original Code: https://github.com/hexiangnan/neural_collaborative_filtering

논문은 0과 1로 user-item interaction으로 matrix을 나타내고 학습했으나,
본 코드에서는 rating을 직접 예측함.
"""

import argparse
import pprint
import torch
from torch import optim
import torch.nn as nn
from data_loader import KMRDDataLoader, BatchIterator

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
        help='Model file name to save. Additional information would be annotated to the file name.'
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
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_loader = KMRDDataLoader(config.data_path)

    batch_iter = BatchIterator(
        data_loader.train_x,
        data_loader.train_y,
        config.batch_size, device)

    for x, y in batch_iter:
        print(x)
        print(x.size())
        print(y)
        print(y.size())
        break


if __name__ == '__main__':
    config = define_argparser()
    main(config)