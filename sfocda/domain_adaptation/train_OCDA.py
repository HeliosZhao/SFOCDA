import os
import sys
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from ..utils.func import adjust_learning_rate
from ..utils.func import loss_calc, pseudo_label_generate
from ..model.deeplab_vgg import DeeplabVGG

def train_source(trainloader, cfg):
    ''' UDA training with source only
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model = DeeplabVGG(cfg, num_classes=cfg.NUM_CLASSES)


    if cfg.TRAIN.START_ITER:
        start_resume = osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{cfg.TRAIN.START_ITER}.pth')
        model.load_state_dict(torch.load(start_resume))
        print(f'====>>> start training from iter {cfg.TRAIN.START_ITER}')
    # saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)

    # model.load_state_dict(saved_state_dict, strict=False)

    model.train()
    model.to(device)

    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.base_params(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)

    trainloader_iter = enumerate(trainloader)
    for i_iter in tqdm(range(cfg.TRAIN.START_ITER, cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()
        model.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        labels = labels[0].unsqueeze(0)
        labels = labels.long().to(device)

        pred_src_main = model.forward_cpss(images_source.cuda(device))
        
        pred_src_main = interp(pred_src_main)
        loss = loss_calc(pred_src_main, labels, device)

        loss.backward()

        if loss > 3:
            continue

        optimizer.step()

        current_losses = {'loss_seg_src': loss}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter > cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()


def train_target(targetloader, cfg):
    ''' UDA training with source only
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model = DeeplabVGG(cfg, num_classes=cfg.NUM_CLASSES)

    saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)

    model.load_state_dict(saved_state_dict, strict=False)
    model.train()
    model.to(device)

    p_model = DeeplabVGG(cfg, num_classes=cfg.NUM_CLASSES)
    p_model.load_state_dict(saved_state_dict, strict=False)
    p_model.eval()
    p_model.to(device)

    cudnn.benchmark = True
    cudnn.enabled = True

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)  ## upsample size = (h,w) / Pillow resize (w,h)
                                
    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    targetloader_iter = enumerate(targetloader)

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training


        _, batch = targetloader_iter.__next__()
        images, images_aug, _, _ = batch
        images = images.to(device)
        images_aug = images_aug.to(device)

        pred = model.forward_cpss(images_aug.cuda(device))

        pred = interp_target(pred)

        with torch.no_grad():
            p_image = images[0].unsqueeze(0) # 1,3,H,W
            outputs = p_model(p_image)
            probs = interp_target(F.softmax(outputs, dim=1)) # 1,19,H,W
            max_val, max_idx = probs.max(dim=1)
            # # generate new labels
            labels = pseudo_label_generate(cfg, max_val, max_idx)


        loss = loss_calc(pred, labels, device)
            
        loss.backward()

        if loss > 3:
            continue
        
        optimizer.step()

        current_losses = {'loss_seg_tgt': loss}

        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            # torch.save(m_model.state_dict(),
            #            osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'Mmodel_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()
