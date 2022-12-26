import numpy as np
import torch
import torch.nn as nn

from .loss import cross_entropy_2d

def pseudo_label_generate(cfg, max_val, max_idx):

    with torch.no_grad():

        if cfg.TRAIN.THRESHOLD and not cfg.TRAIN.PERCENT:
            max_idx[max_val < cfg.TRAIN.THRESHOLD] = 255

        elif cfg.TRAIN.PERCENT and not cfg.TRAIN.THRESHOLD:
            for ii in range(cfg.NUM_CLASSES):
                cls_max_val = max_val.clone()
                cls_max_val[max_idx != ii] = 0
                cls_max_val = cls_max_val.reshape(max_idx.size(0), -1)
                for jj in range(max_idx.size(0)):
                    num_cls = len(torch.nonzero(cls_max_val[jj]))
                    cls_topk = int(num_cls * cfg.TRAIN.PERCENT)
                    if cls_topk == 0:
                        continue
                    cls_top_val, cls_top_idx = torch.kthvalue(- cls_max_val[jj], cls_topk, 0, True)
                    sel_top_idx = cls_max_val[jj] < - cls_top_val
                    sel_top_idx = sel_top_idx.reshape(max_idx.size(1), max_idx.size(2))
                    sel_cls_idx = (max_idx[jj, :, :] == ii)
                    max_idx[jj, :, :][sel_top_idx * sel_cls_idx] = 255
                    
        elif cfg.TRAIN.PERCENT and cfg.TRAIN.THRESHOLD:
            for ii in range(cfg.NUM_CLASSES):
                cls_max_val = max_val.clone()
                cls_max_val[max_idx != ii] = 0
                cls_max_val = cls_max_val.reshape(max_idx.size(0), -1)
                for jj in range(max_idx.size(0)):
                    num_cls = len(torch.nonzero(cls_max_val[jj]))
                    cls_topk = int(num_cls * cfg.TRAIN.PERCENT)
                    if cls_topk == 0:
                        continue
                    cls_top_val, cls_top_idx = torch.kthvalue(- cls_max_val[jj], cls_topk, 0, True) 
                    class_threshold = min(-cls_top_val, cfg.TRAIN.THRESHOLD)
                    sel_top_idx = cls_max_val[jj] < class_threshold ## selected as 255
                    sel_top_idx = sel_top_idx.reshape(max_idx.size(1), max_idx.size(2))
                    sel_cls_idx = (max_idx[jj, :, :] == ii)
                    max_idx[jj, :, :][sel_top_idx * sel_cls_idx] = 255

        else:
            return max_idx.long()
        
        labels = max_idx.long()

    return labels



def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    return cross_entropy_2d(pred, label)

def center_loss(features, label, mem_features):
    '''
    features B,C,H,W --> upsampled and normalized
    label B,H,W
    mem_features num_classes,C
    '''
    B,C,H,W = features.size()
    features = features.permute(0,2,3,1).reshape(-1, C) # BHW,C

    label = label.reshape(-1) # BHW
    num_classes = mem_features.size(0)

    source_features = torch.zeros_like(features) # BHW,C

    for i in range(num_classes):
        features_per_class = features[label==i]
        source_features[label==i] = mem_features[i].expand(features_per_class.size(0), C)

    return nn.MSELoss()(features, source_features)

def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def _adjust_learning_rate_together(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr

def adjust_learning_rate_together(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate_together(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)

def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
