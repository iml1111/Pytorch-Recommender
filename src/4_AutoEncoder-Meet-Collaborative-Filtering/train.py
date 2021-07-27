import argparse
from pprint import pprint
import torch
from torch import optim
import torch.nn as nn
from model import AutoEncoder
from data_loader import KMRDDataset, DataLoader
from trainer import Trainer

KMRD_SMALL_DATA_PATH = "../data/kmrd/kmr_dataset/datafile/kmrd-small/rates.csv"
KMRD_2M_DATA_PATH = "../data/kmrd/kmr_dataset/datafile/kmrd/rates-2m.csv"

def define_argparser():

    p = argparse.ArgumentParser()
    p.add_argument(
        '--model_fn',
        default='./model.pth',
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--data_path',
        default=KMRD_SMALL_DATA_PATH,
        help='Dataset Path, Default=%(default)s'
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=30,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Train data ratio. Default=%(default)s'
    )
    p.add_argument(
        '--valid_ratio',
        type=float,
        default=0.05,
        help='Valid data ratio. Default=%(default)s'
    )
    p.add_argument(
        '--act_func',
        type=str,
        default="elu",
        help='Activation Function. Default=%(default)s'
    )
    config = p.parse_args()
    return config


def main(config):
    print("# Config")
    pprint(vars(config))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", torch.cuda.get_device_name(0))
    dataset = KMRDDataset(config.data_path)

    train_size = int(len(dataset) * config.train_ratio)
    valid_size = int(len(dataset) * config.valid_ratio)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = (
        torch.utils.data.random_split(
            dataset, (train_size, valid_size, test_size)
        )
    )
    print("Train:", train_size)
    print("Valid:", valid_size)

    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    model = AutoEncoder(len(dataset.user2idx), config.act_func).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss().to(device)

    trainer = Trainer(model, optimizer, crit, device)
    trainer.train(train_data_loader, valid_data_loader, config)

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