"""
https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html
https://github.com/jrzaurin/pytorch-widedeep

Warning
https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
"""

import argparse
from pprint import pprint
import numpy as np
from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy

from data_loader import KMRDDataLoader

KMRD_SMALL_DATA_PATH = "../data/kmrd/kmr_dataset/datafile/kmrd-small"



def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument(
        '--data_path',
        default=KMRD_SMALL_DATA_PATH,
        help='Dataset Path, Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=16,
        help='Embedding Latent Vector Size. Default=%(default)s'
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=30,
        help='Number of epochs to train. Default=%(default)s'
    )
    config = p.parse_args()
    return config


def main(config):
    print("# Config")
    pprint(vars(config))

    data_loader = KMRDDataLoader(config.data_path)
    print("Train:", data_loader.train_df.shape)

    # wide
    wide_preprocessor = WidePreprocessor(
        wide_cols=data_loader.wide_cols, 
        crossed_cols= data_loader.cross_cols
    )
    X_wide = wide_preprocessor.fit_transform(data_loader.train_df)
    wide = Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)

    # deeptabular
    tab_preprocessor = TabPreprocessor(
        embed_cols=data_loader.embed_cols, 
        continuous_cols=data_loader.continuous_cols
    )
    X_tab = tab_preprocessor.fit_transform(data_loader.train_df)
    deeptabular = TabMlp(
        mlp_hidden_dims=[64, 32],
        column_idx=tab_preprocessor.column_idx,
        embed_input=tab_preprocessor.embeddings_input,
        continuous_cols=data_loader.continuous_cols,
    )

    model = WideDeep(wide=wide, deeptabular=deeptabular)

    trainer = Trainer(model, objective='binary', metrics=[Accuracy])
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=data_loader.target,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        val_split=0.1,
    )


if __name__ == '__main__':
    config = define_argparser()
    main(config)

    
    
