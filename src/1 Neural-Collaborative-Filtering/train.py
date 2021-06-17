"""
# Neural Collaborative Filtering
Paper: https://arxiv.org/pdf/1708.05031.pdf
Original Code: https://github.com/hexiangnan/neural_collaborative_filtering

논문은 0과 1로 user-item interaction으로 matrix을 나타내고 학습했으나, 본 코드에서는 rating을 직접 예측함.
"""

import argparse
import pprint
import torch
from torch import optim
import torch.nn as nn


def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        default='./models/model.pth',
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
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
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=32,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )


def main(config):
    pass


if __name__ == '__main__':
    config = define_argparser()
    main(config)