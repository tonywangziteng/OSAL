import json
import multiprocessing as mp
import os
import random
import pdb
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset_utils import get_dataloader
from model.OsalModel import OsalModel
from utils.nms_utils import soft_nms_proposal
import os.path as osp
from collections import OrderedDict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_origin_map():
    origin_map = []
    for feature_len in [50, 25, 13, 7, 4]:
        mapping = np.arange(feature_len)
        time_gap = 100//feature_len
        mapping = mapping*time_gap + time_gap//2
        origin_map.append(mapping/100.)
    return origin_map


if __name__ == "__main__":
    config_path = './config/config.json'
    try:
        f = open(config_path)
        config = json.load(f)
    except IOError:
        print('Model Building Error: errors occur when loading config file from '+config_path)
    
    model = OsalModel().to(device)

    # load_model
    weight_dir = config['checkpoint_dir']
    weight_path = osp.join(weight_dir, 'weight1.pth.tar')
    checkpoint = torch.load(weight_path)

    model.load_state_dict(checkpoint['state_dict'])

    print('model weight loaded')

    # get test dataloader
    test_loader = get_dataloader(config, 'testing', batch_size=1)
    print('testing dataset loaded')

    print('start testing, save result to a json_file')
    # create a json file to save the result before nms
    data_before_nms = {}
    f = open('data_before_nms.json','w',encoding='utf-8')

    pbar = tqdm.tqdm(test_loader)
    mapping = get_origin_map()
    for features, cls_gt_list, duration_list, video_name_list in pbar:
        features = features.float().to(device)
        cls_list, reg_list, cls_list_final, reg_list_final = model(features)
        raw_result = {}
        raw_result['video_name'] = video_name_list[0]
        # cls_ds_list, reg_ds_list = cls_list[0], reg_ds_list[0]
        # cls_us_list, reg_us_list = cls_list_final[0], reg_list_final[0]
        for layer_index in range(len(cls_list)):
            # get raw result in this layer
            cls_ds = cls_list[layer_index][0].detach().cpu().numpy()
            reg_ds = np.clip(reg_list[layer_index][0].detach().cpu().numpy(), 0, 1) 
            cls_us = cls_list_final[layer_index][0].detach().cpu().numpy()
            reg_us = np.clip(reg_list_final[layer_index][0].detach().cpu().numpy(), 0, 1) 

            cls_result = np.power(cls_ds * cls_us, 0.5) 
            reg_result = reg_ds[0:2] * reg_us[0:2]
            np.random.rand()
            # get fg_indice
            fg_result = np.power(cls_result[200] * np.power(reg_ds[2] * reg_ds[2], 0.5), 0.5) 
            fg_indice = np.argwhere(fg_result > config['bg_threshold'])

            # get positive result
            positive_cls = cls_result[:200, fg_indice[:, 0]]
            positive_reg = reg_result[:, fg_indice[:, 0]]

            base_point = mapping[layer_index][fg_indice[:, 0]]
            start_bias = -reg_ds[0, fg_indice[:, 0]]
            end_bias = reg_ds[1, fg_indice[:, 0]]
            start_delta = reg_us[0, fg_indice[:, 0]]
            end_delta = reg_us[1, fg_indice[:, 0]]
            start = base_point + start_bias + start_delta
            end = base_point + end_bias + end_delta

            print('ground truth duration', duration_list)
            print('predicted durations: ', np.stack([start, end], axis=-1))
            print('ground_truth class', cls_gt_list)
            print('predicted class ', np.argmax(positive_cls, 0))
            pdb.set_trace()

    data_before_nms = json.dumps(data_before_nms)
    f.write(data_before_nms)
    f.close()
