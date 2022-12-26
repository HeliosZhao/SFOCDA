# --------------------------------------------------------
# Configurations for domain adaptation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py
# --------------------------------------------------------

import os.path as osp

import numpy as np
from easydict import EasyDict

from ..utils import project_root
from ..utils.serialization import yaml_load


cfg = EasyDict()

# COMMON CONFIGS
# source domain
cfg.SOURCE = 'GTA'
# target domain
cfg.TARGET = 'Cityscapes'
# Number of workers for dataloading
cfg.NUM_WORKERS = 4
cfg.ARCH = 'DeepLab'
# List of training images
cfg.DATA_LIST_SOURCE = str(project_root / 'sfocda/dataset/gta5_list/train.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'sfocda/dataset/cityscapes_list/{}.txt')
cfg.DATA_LIST_TARGET_EASY = ''
cfg.DATA_LIST_TARGET_HARD = ''
# Directories
cfg.DATA_DIRECTORY_SOURCE = '../../data/C-Driving/train/source'
cfg.DATA_DIRECTORY_TARGET = str(project_root / 'data/Cityscapes')
# Number of object classes
cfg.NUM_CLASSES = 19
# Exp dirs
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / 'experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.GPU_ID = 0

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'all'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 720)
cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
# Class info
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.INFO_TARGET = str(project_root / 'sfocda/dataset/cityscapes_list/info.json')
# Segmentation network params
cfg.TRAIN.MODEL = 'VGG'
cfg.TRAIN.MULTI_LEVEL = False
cfg.TRAIN.RESTORE_FROM = ''
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.WITH_CLASSIFIER_LR = False
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1  # weight of conv4 prediction. Used in multi-level setting.
## optimizer
cfg.TRAIN.FREEZE_CLASSIFIER = False
cfg.TRAIN.BASE_PARAMS = True

# Domain adaptation
cfg.TRAIN.DA_METHOD = 'train_source'
# Adversarial training params
cfg.TRAIN.LEARNING_RATE_D = 1e-4
cfg.TRAIN.LAMBDA_ADV_MAIN = 0.001
cfg.TRAIN.LAMBDA_ADV_AUX = 0.0002
# MinEnt params
cfg.TRAIN.LAMBDA_ENT_MAIN = 0.001
cfg.TRAIN.LAMBDA_ENT_AUX = 0.0002
# Other params
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.START_ITER = 0
cfg.TRAIN.EARLY_STOP = 150000
cfg.TRAIN.SAVE_PRED_EVERY = 1000
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 100

## pseudo labels
cfg.TRAIN.PERCENT = 0.
cfg.TRAIN.THRESHOLD = 0.

## CPSS
cfg.TRAIN.PROB = 0.3
cfg.TRAIN.NUM_H = 2
cfg.TRAIN.NUM_W = 2

## SCE LOSS
cfg.TRAIN.LOSS = 'CE'
cfg.TRAIN.ALPHA = 0.1
cfg.TRAIN.BETA = 1.0
cfg.TRAIN.CURRICULUM = False
cfg.TRAIN.ADV_LOSS = 'SCE'

# Augmentation
cfg.TRAIN.COLOR_JITTER = 0.5
cfg.TRAIN.GAUSSIAN_BLUR = 0.5
cfg.TRAIN.GRAY_SCALE = 0.2

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'best'  # {'single', 'best'}
cfg.TEST.SAVE_CKPT = True
# model
cfg.TEST.MODEL = ('VGG',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 1000  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 150000  # used in 'best' mode
# Test sets
cfg.TEST.SET_TARGET = 'val'
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (1280, 720)
cfg.TEST.INFO_TARGET = str(project_root / 'sfocda/dataset/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = True
cfg.TEST.DATA_LIST_TARGET = 'sfocda/dataset/bdd_list/val/3domains.txt'
cfg.TEST.DATA_DIRECTORY_TARGET = ''


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            print('old : ',b[k], 'new : ',v)
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
