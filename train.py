# --------------------------------------------------------
# sfocda training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import random
import warnings

import numpy as np
import yaml
import torch
from torch.utils import data
import torchvision.transforms as transforms
from sfocda.dataset.gta5 import GTA5DataSet, GTA5_Aug_DataSet
from sfocda.domain_adaptation.config import cfg, cfg_from_file
from sfocda.dataset.bdd_dataset import BDDDataSet_Aug
from sfocda.dataset.augmentation import GaussianBlur
from sfocda.domain_adaptation.train_OCDA import *

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('sfocda_DRY_RUN', '0') == '1':
        return


    augmentation = []
    if cfg.TRAIN.COLOR_JITTER:
        augmentation += [transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=cfg.TRAIN.COLOR_JITTER)]
    if cfg.TRAIN.GAUSSIAN_BLUR:
        augmentation += [transforms.RandomApply([GaussianBlur([.1, 2.])], p=cfg.TRAIN.GAUSSIAN_BLUR)]
    if cfg.TRAIN.GRAY_SCALE:
        augmentation += [transforms.RandomApply([
            transforms.Grayscale(num_output_channels=3)  # not strengthened
        ], p=cfg.TRAIN.GRAY_SCALE)]


    # DATALOADERS
    if len(augmentation):
        print('************ using photometric augmentations ******************')
        source_dataset = GTA5_Aug_DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                    list_path=cfg.DATA_LIST_SOURCE,
                                    set=cfg.TRAIN.SET_SOURCE,
                                    max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                    mean=cfg.TRAIN.IMG_MEAN,
                                    augmentations=transforms.Compose(augmentation))
    else:
        print('------>>> without photometric augmentations ******************')
        source_dataset = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                    list_path=cfg.DATA_LIST_SOURCE,
                                    set=cfg.TRAIN.SET_SOURCE,
                                    max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                    mean=cfg.TRAIN.IMG_MEAN)

    source_loader = data.DataLoader(source_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    target_dataset = BDDDataSet_Aug(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN,
                                       augmentation=transforms.Compose(augmentation))

    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    if cfg.TRAIN.DA_METHOD == 'train_source':
        train_source(source_loader, cfg)

    
    elif cfg.TRAIN.DA_METHOD == 'train_target':
        train_target(target_loader, cfg)



if __name__ == '__main__':
    main()
