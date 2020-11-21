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

# GPU setting.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    

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

    perceptive_fields = config['perceptive_fields']
    perceptive_fields = np.array(perceptive_fields)/200.
    
    model = OsalModel().to(device)

    # load_model
    weight_dir = config['checkpoint_dir']
    weight_path = osp.join(weight_dir, 'final_0/epoch5_3.262289514799812_adam_param.pth.tar')
    checkpoint = torch.load(weight_path)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # model.down_sample_head.scales.state_dict()
    #model.down_sample_head.scales

    print('model weight loaded')

    # get test dataloader
    test_loader = get_dataloader(config, 'testing', batch_size=1, shuffle=False)
    print('testing dataset loaded')

    print('start testing, save result to a json_file')
    # create a json file to save the result before nms
    data_before_nms = {}
    f = open('data_before_nms.json','w',encoding='utf-8')

    pbar = tqdm.tqdm(test_loader)
    mapping = get_origin_map()
    for features, cls_gt_list, duration_list, video_name_list in pbar:
        features = features.float().to(device)
        cls_list, reg_list, start_end = model(features)
        raw_result = {}
        raw_result['video_name'] = video_name_list[0]
        start_end = start_end[0].detach().cpu().numpy()

        for layer_index in range(len(cls_list)):
            # get raw result in this layer
            cls_ds = cls_list[layer_index][0].detach().cpu().numpy()
            reg_ds = np.clip(reg_list[layer_index][0].detach().cpu().numpy(), 0, 1) 

            # cls_us = cls_list_final[layer_index][0].detach().cpu().numpy()
            # reg_us = np.clip(reg_list_final[layer_index][0].detach().cpu().numpy(), 0, 1) 

            # cls_result = np.power(cls_ds * cls_us, 0.5) 
            cls_result = cls_ds

            fg_result = cls_result[200]
            start_bias = reg_ds[0]
            end_bias = reg_ds[1]

            base_point = mapping[layer_index]
            start = base_point - start_bias
            start = np.clip(start, 0, 1)
            end = base_point + end_bias
            end = np.clip(end, 0, 1)

            start_idx_floor = np.floor(start*100).astype(np.int)
            start_idx_floor[start_idx_floor==100] = 99
            start_score = start_end[0, start_idx_floor]
            
            end_idx_floor = np.floor(end*100).astype(np.int)
            end_idx_floor[end_idx_floor==100] = 99
            end_score = start_end[1, end_idx_floor]

            fg_result = fg_result * np.sqrt(start_score * end_score)
# 
            fg_indice = \
                (np.max(reg_ds[:2, :], 0)> perceptive_fields[layer_index]) & \
                (np.max(reg_ds[:2, :], 0)< perceptive_fields[layer_index+1]) &\
                (fg_result > config['bg_threshold'])

            fg_indice = fg_indice.nonzero()

            positive_cls = cls_result[:200, fg_indice]
            if positive_cls.shape[-1]>0:
                positive_cls = positive_cls.argmax(0)

            start = start[fg_indice] # + start_delta
            end = end[fg_indice] # + end_delta
            print('ground truth duration', duration_list)
            print('predicted durations: ', np.stack([start, end], axis=-1))
            print('ground_truth class', cls_gt_list)
            print('predicted class ', positive_cls)
            print('predicted score ', fg_result[fg_indice])
            # pdb.set_trace()
        pdb.set_trace()

    data_before_nms = json.dumps(data_before_nms)
    f.write(data_before_nms)
    f.close()
