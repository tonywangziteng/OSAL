import json
import multiprocessing as mp
import os
import random
import pdb
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset_utils import get_dataloader
from model.OsalModel import OsalModel
from utils.nms_utils import soft_nms_proposal

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "1"     

def get_origin_map():
        origin_map = []
        feature_lens = config['feature_lens']
        for feature_len in feature_lens:
            mapping = np.arange(feature_len)
            time_gap = 100//feature_len   #该层每两段之间的长度
            mapping = mapping*time_gap + time_gap//2
            origin_map.append(mapping/100.)
        return origin_map    #每层中间的对应时间（0，1）

def standarize(input_array):
    miu = input_array.mean()
    std = input_array.std()
    return (input_array-miu)/std

if __name__=="__main__":

    config_path = './config/config.json'
    try:
        f = open(config_path)
        config = json.load(f)
    except IOError:
        print('Model Building Error: errors occur when loading config file from '+config_path)
    
    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])
    
    print("Load the model.")
    model = OsalModel()
    model = model.cuda()
    feature_lens = config["feature_lens"]

    origin_map = get_origin_map() 
    for i in range(len(origin_map)):
        origin_map[i] = np.array(origin_map[i])
    origin_map_extended = np.concatenate(origin_map)

    perceptive_fields = config["perceptive_fields"]
    perceptive_fields = np.array(perceptive_fields) / 200.
    weight_dir = config['checkpoint_dir']
    print(weight_dir)
    weight_path = osp.join(weight_dir, 'iou_cheat2/epoch9_17.220708085397376_adam_param.pth.tar')
    print(weight_path)
    checkpoint = torch.load(weight_path)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_dataloader = get_dataloader(config, 'testing', batch_size=1, shuffle=False)
    result_json_path = './output/result_proposal' + '.json'
    prd_res_ = []

    print("Process data and save.")
    with torch.no_grad():
        model.eval()
        for idx, (raw_feature, cls_gt, duration_list, video_name) in enumerate(test_dataloader): 
            raw_feature = raw_feature.cuda()
            cls_list_dn_raw, reg_list_dn_raw,  = model(raw_feature)        
            
            prd_res = [] # initialize the prediction list

            # 生成每个动作的左右感受野边界, shape:(99, 2)
            perceptive_bounds = []
            for layer_index in range(len(cls_list_dn_raw)):
                feature_len = feature_lens[layer_index]
                pb = np.array([perceptive_fields[layer_index], perceptive_fields[layer_index+1]])
                pb_expanded = np.repeat(pb[np.newaxis, :], feature_len, 0)
                perceptive_bounds.append(pb_expanded)
            perceptive_bounds_expanded = np.concatenate(perceptive_bounds, 0)

            # 不同层concat到一起
            cls_expanded = torch.cat(cls_list_dn_raw, -1).squeeze(0).detach().cpu().numpy()
            reg_expanded = torch.cat(reg_list_dn_raw, -1).squeeze(0).detach().cpu().numpy()
                
            '''Find background and foreground'''
            cls_result = cls_expanded[:200] # 多分类
            fg_result = cls_expanded[200]# * reg_expanded[2] # 二分类 * centerness
            fg_result_regularized = standarize(fg_result)
            # pdb.set_trace()
            # fg_result = cls_expanded[200] * reg_expanded[2] * cls_result.max(0)
            reg_result = reg_expanded[:2]

            fg_pos = \
                (np.max(reg_result, axis=0) > perceptive_bounds_expanded[:, 0]) & \
                (np.max(reg_result, axis=0) < perceptive_bounds_expanded[:, 1]) & \
                (fg_result_regularized > config['bg_threshold'])
            fg_indice = fg_pos.nonzero()

            cls_positive = cls_result[:, fg_pos]
            cls_each = np.argmax(cls_positive, axis=0)
            idx_each = fg_indice[0]

            # 把所有的正样本放到proposals这个list里面
            proposals = []
            for i, i_pos in enumerate(idx_each):
                xmin = np.clip(origin_map_extended[i_pos]-reg_result[0, i_pos], 0, 1)
                xmax = np.clip(origin_map_extended[i_pos]+reg_result[1, i_pos], 0, 1)   #满足条件的时候把偏移量加上

                # score = reg_list[2, i]  #centerness
                # score = fg_result[i_pos] # foreground and background
                score = fg_result_regularized[i_pos]
                cls_idx = cls_each[i]

                proposals.append([xmin, xmax, score, cls_idx]) #根据index_each给每个位置打分
                            
            alpha = 0.5
            t1 = 0.5 # lower threshold
            t2 = 0.002 # higher threshold
            # proposals = torch.from_numpy(proposals)
            columns = ["xmin", "xmax", "score" ,"cls_idx"]
            proposals = pd.DataFrame(proposals, columns=columns)
            # pdb.set_trace()
            nms_res = soft_nms_proposal(proposals, alpha, t1, t2)

            """Save information to prd_res_"""
            for i in range(len(nms_res['xmin'])):
                prd_res.append([video_name[0], nms_res['xmin'][i], nms_res['xmax'][i], nms_res['score'][i], int(nms_res['cls_idx'][i])])
            
            print("length of prd_res for video %d :" % idx, len(prd_res))
            if not len(prd_res):
                continue
            prd_res_.append({"%s"%video_name[0]:prd_res})
            # pdb.set_trace()
            
        with open(result_json_path, 'w') as j:
            json.dump(prd_res_, j)
        print('Already saved the json, waiting for evaluation.')
        pdb.set_trace()
        # map


            






    
