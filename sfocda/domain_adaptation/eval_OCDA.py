# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------

import os.path as osp
import time
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..utils.func import per_class_iu, fast_hist
from ..utils.serialization import pickle_dump, pickle_load
from ..model.deeplab_vgg import DeeplabVGG

def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=False,
                                verbose=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    elif cfg.TEST.MODE == 'best_accumulate':
        eval_best_accumulate(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")

def evaluate_domain_adaptation_bdd(models, test_loader, domain_list, cfg,
                                fixed_test_size=False,
                                verbose=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single_bdd(cfg, domain_list, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best_bdd(cfg, domain_list, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    elif cfg.TEST.MODE == 'best_accumulate':
        eval_best_accumulate_bdd(cfg, domain_list, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.cuda(device)) #[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    # if verbose:
    #     display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
                ## wait for save
                time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                image, label, _, name = next(test_iter)
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device)) #[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)
                    # print('output',output.shape)
                    # print('label', label.shape)
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        # if verbose:
        #     display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)

def eval_single_bdd(cfg, domain_list, models,
                device, test_loader_all, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    mIoU_list = {}
    hist_list = {}
    inters_over_union_classes_list = {}
    co_mIoU = 0

    hist_3domains = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for d_name in domain_list:
        if d_name == '3domains':
            continue
        test_loader = test_loader_all[d_name]
        # eval
        hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
        # for index, batch in enumerate(test_loader):
        #     image, _, _, name = batch
        test_iter = iter(test_loader)
        for index in tqdm(range(len(test_loader))):
            image, label, _, name = next(test_iter)
            if not fixed_test_size:
                interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
            with torch.no_grad():
                pred_main = models[0](image.cuda(device))  # [1]
                output = interp(pred_main).cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.argmax(output, axis=2)
                # print('output',output.shape)
                # print('label', label.shape)
            label = label.numpy()[0]
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
            # if verbose and index > 0 and index % 100 == 0:
            #     print('{:d} / {:d}: {:0.2f}'.format(
            #         index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
        inters_over_union_classes = per_class_iu(hist)

        hist_list[d_name] = hist
        # save for all settings
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        print (computed_miou)
        mIoU_list[d_name] = computed_miou
        if d_name != '3domains':
            co_mIoU += mIoU_list[d_name]
            if d_name != 'overcast':
                hist_3domains += hist_list[d_name]

    inters_over_union_classes = per_class_iu(hist_3domains)
    computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    mIoU_list['3domains'] = computed_miou
    mIoU_list['co_mIoU'] = round(co_mIoU / 4, 2)
    print('\tCurrent mIoU:', mIoU_list)


def eval_best_bdd(cfg, domain_list, models,
              device, test_loader_all, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res_{}_{}.pkl'.format(str(cfg.TEST.INPUT_SIZE_TARGET[0]), str(cfg.TEST.INPUT_SIZE_TARGET[1])))
    print (cache_path)
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''
    for i_iter in range(start_iter, max_iter + 1, step):
        # restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'GTA5_{i_iter}.pth')
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        
        mIoU_list = {}
        hist_list = {}
        inters_over_union_classes_list = {}
        co_mIoU = 0
        if i_iter not in all_res.keys():

            if not osp.exists(restore_from):
                # continue
                if cfg.TEST.WAIT_MODEL:
                    print('Waiting for model..!')
                    while not osp.exists(restore_from):
                        time.sleep(5)
                    # Allow time to save model
                    time.sleep(5)
            print("Evaluating model", restore_from)

            load_checkpoint_for_evaluation(models[0], restore_from, device)
            hist_3domains = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            for d_name in domain_list:
                if d_name == '3domains':
                    continue
                test_loader = test_loader_all[d_name]
                # eval
                hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
                # for index, batch in enumerate(test_loader):
                #     image, _, _, name = batch
                test_iter = iter(test_loader)
                for index in tqdm(range(len(test_loader))):
                    image, label, _, name = next(test_iter)
                    if not fixed_test_size:
                        interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                    with torch.no_grad():
                        pred_main = models[0](image.cuda(device))  # [1]
                        pred_main = interp(pred_main).cpu()
                        for t in range(pred_main.size(0)):
                            output = pred_main.data[t].numpy()
                            output = output.transpose(1, 2, 0)
                            output = np.argmax(output, axis=2)
                            # print('output',output.shape)
                            # print('label', label.shape)
                            # print('label type : ', type(label))
                            label_per = label.numpy()[t]
                            hist += fast_hist(label_per.flatten(), output.flatten(), cfg.NUM_CLASSES)
                    # if verbose and index > 0 and index % 100 == 0:
                    #     print('{:d} / {:d}: {:0.2f}'.format(
                    #         index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
                inters_over_union_classes = per_class_iu(hist)

                hist_list[d_name] = hist
                # save for all settings
                computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
                mIoU_list[d_name] = computed_miou
                if d_name != '3domains':
                    co_mIoU += mIoU_list[d_name]
                    if d_name != 'overcast':
                        hist_3domains += hist_list[d_name]
            all_res[i_iter] = hist_list
            pickle_dump(all_res, cache_path)
        else:
            print("Evaluating model", restore_from)
            hist_list = all_res[i_iter]
            hist_3domains = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            for d_name in domain_list:
                # save for all settings
                if d_name == '3domains':
                    continue
                inters_over_union_classes = per_class_iu(hist_list[d_name])
                computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
                mIoU_list[d_name] = computed_miou
                if d_name != '3domains':
                    co_mIoU += mIoU_list[d_name]
                    if d_name != 'overcast':
                        hist_3domains += hist_list[d_name]

        inters_over_union_classes = per_class_iu(hist_3domains)
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        mIoU_list['3domains'] = computed_miou
        mIoU_list['co_mIoU'] = round(co_mIoU / 4, 2)
        if cur_best_miou < mIoU_list['co_mIoU']:
            cur_best_miou = mIoU_list['co_mIoU']
            cur_best_mioulist = mIoU_list
            cur_best_model = restore_from
        print('\tCurrent mIoU:', mIoU_list)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_mioulist)

def eval_best_accumulate_bdd(cfg, domain_list, models,
              device, test_loader_all, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res_{}_{}.pkl'.format(str(cfg.TEST.INPUT_SIZE_TARGET[0]), str(cfg.TEST.INPUT_SIZE_TARGET[1])))
    print (cache_path)
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''

    accumulate_model = DeeplabVGG(cfg, num_classes=cfg.NUM_CLASSES)
    accumulate_model.eval()
    accumulate_model.cuda()

    accumulate_model.load_state_dict(torch.load(cfg.TRAIN.RESTORE_FROM), strict=False)

    num_accumulate = 0

    accumulate_save = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'accumulate_model')

    if not osp.exists(accumulate_save):
        os.makedirs(accumulate_save)

    for i_iter in range(start_iter, max_iter + 1, step):
        # restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'GTA5_{i_iter}.pth')
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        
        mIoU_list = {}
        hist_list = {}
        inters_over_union_classes_list = {}
        co_mIoU = 0
        if i_iter not in all_res.keys():

            if not osp.exists(restore_from):
                # continue
                if cfg.TEST.WAIT_MODEL:
                    print('Waiting for model..!')
                    while not osp.exists(restore_from):
                        time.sleep(5)
                    # Allow time to save model
                    time.sleep(5)
            print("Evaluating model", restore_from)

            load_checkpoint_for_evaluation(models[0], restore_from, device)
            num_accumulate += 1
            accumulate_model = momentum_update_key_encoder(accumulate_model, models[0], num_accumulate)
            if cfg.TEST.SAVE_CKPT:
                torch.save(accumulate_model.state_dict(),
                       osp.join(accumulate_save, f'model_{i_iter}.pth'))
            hist_3domains = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            for d_name in domain_list:
                if d_name == '3domains':
                    continue
                test_loader = test_loader_all[d_name]
                # eval
                hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
                # for index, batch in enumerate(test_loader):
                #     image, _, _, name = batch
                test_iter = iter(test_loader)
                for index in tqdm(range(len(test_loader))):
                    image, label, _, name = next(test_iter)
                    if not fixed_test_size:
                        interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                    with torch.no_grad():
                        pred_main = accumulate_model(image.cuda(device))  # [1]
                        pred_main = interp(pred_main).cpu()
                        for t in range(pred_main.size(0)):
                            output = pred_main.data[t].numpy()
                            output = output.transpose(1, 2, 0)
                            output = np.argmax(output, axis=2)
                            # print('output',output.shape)
                            # print('label', label.shape)
                            # print('label type : ', type(label))
                            label_per = label.numpy()[t]
                            hist += fast_hist(label_per.flatten(), output.flatten(), cfg.NUM_CLASSES)
                    # if verbose and index > 0 and index % 100 == 0:
                    #     print('{:d} / {:d}: {:0.2f}'.format(
                    #         index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
                inters_over_union_classes = per_class_iu(hist)

                hist_list[d_name] = hist
                # save for all settings
                computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
                mIoU_list[d_name] = computed_miou
                if d_name != '3domains':
                    co_mIoU += mIoU_list[d_name]
                    if d_name != 'overcast':
                        hist_3domains += hist_list[d_name]
            all_res[i_iter] = hist_list
            pickle_dump(all_res, cache_path)
        else:
            print("Evaluating model", restore_from)
            hist_list = all_res[i_iter]
            hist_3domains = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            for d_name in domain_list:
                # save for all settings
                if d_name == '3domains':
                    continue
                inters_over_union_classes = per_class_iu(hist_list[d_name])
                computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
                mIoU_list[d_name] = computed_miou
                if d_name != '3domains':
                    co_mIoU += mIoU_list[d_name]
                    if d_name != 'overcast':
                        hist_3domains += hist_list[d_name]

        inters_over_union_classes = per_class_iu(hist_3domains)
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        mIoU_list['3domains'] = computed_miou
        mIoU_list['co_mIoU'] = round(co_mIoU / 4, 2)
        if cur_best_miou < mIoU_list['co_mIoU']:
            cur_best_miou = mIoU_list['co_mIoU']
            cur_best_mioulist = mIoU_list
            cur_best_model = restore_from
        print('\tCurrent mIoU:', mIoU_list)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_mioulist)



def momentum_update_key_encoder(model_ema, model_current, num=1):

    for param_q, param_k in zip(model_ema.parameters(), model_current.parameters()):
        param_q.data = (param_q.data * num + param_k.data) / (num + 1)

    return model_ema

def eval_best_accumulate(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''

    accumulate_model = DeeplabVGG(cfg, num_classes=cfg.NUM_CLASSES)
    accumulate_model.eval()
    accumulate_model.cuda()

    accumulate_model.load_state_dict(torch.load(cfg.TRAIN.RESTORE_FROM), strict=False)

    num_accumulate = 0

    accumulate_save = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'accumulate_model')

    if not osp.exists(accumulate_save):
        os.makedirs(accumulate_save)

    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
                ## wait for save
                time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            num_accumulate += 1
            accumulate_model = momentum_update_key_encoder(accumulate_model, models[0], num_accumulate)
            if cfg.TEST.SAVE_CKPT:
                torch.save(accumulate_model.state_dict(),
                       osp.join(accumulate_save, f'model_{i_iter}.pth'))
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            # for index, batch in enumerate(test_loader):
            #     image, _, _, name = batch
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                image, label, _, name = next(test_iter)
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                # with torch.no_grad():
                #     pred_main = accumulate_model(image.cuda(device)) #[1]
                #     output = interp(pred_main).cpu().data[0].numpy()
                #     output = output.transpose(1, 2, 0)
                #     output = np.argmax(output, axis=2)
                #     # print('output',output.shape)
                #     # print('label', label.shape)
                # label = label.numpy()[0]
                # hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                # if verbose and index > 0 and index % 100 == 0:
                #     print('{:d} / {:d}: {:0.2f}'.format(
                #         index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))

                with torch.no_grad():
                    pred_main = accumulate_model(image.cuda(device)) #[1] #B
                    pred_main = interp(pred_main).cpu()
                    for t in range(pred_main.size(0)):
                        output = pred_main.data[t].numpy()
                        output = output.transpose(1, 2, 0)
                        output = np.argmax(output, axis=2)
                        # print('output',output.shape)
                        # print('label', label.shape)
                        # print('label type : ', type(label))
                        label_per = label.numpy()[t]
                        hist += fast_hist(label_per.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % (100//pred_main.size(0)) == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))


            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        # if verbose:
        #     display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)



def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict, strict=False)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
