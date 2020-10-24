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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     

def get_origin_map():
        origin_map = []
        feature_lens = config['feature_lens']
        for feature_len in feature_lens:
            mapping = np.arange(feature_len)
            time_gap = 100//feature_len
            mapping = mapping*time_gap + time_gap//2
            origin_map.append(mapping/100.)
        return origin_map

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

    weight_dir = config['checkpoint_dir']
    weight_path = osp.join(weight_dir, 'weight1.pth.tar')
    checkpoint = torch.load(weight_path)

    model.load_state_dict(checkpoint['state_dict'])


    #checkpoint = torch.load(opt.checkpoint_path + '9_param.pth.tar')
    #model.load_state_dict(checkpoint['satate_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #start_epoch = checkpoint['epoch']

    test_dataloader = get_dataloader(config, 'testing', batch_size=1)
    result_json_path = './output/result_proposal' + '.json'
    prd_res_ = []

    print("Process data and save.")
    with torch.no_grad():
        model.eval()
        for idx, (raw_feature, cls_gt, duration_list, video_name) in enumerate(test_dataloader): 
            raw_feature = raw_feature.cuda()
            cls_list_dn_raw, reg_list_dn_raw, cls_list_up_raw, reg_list_up_raw,  = model(raw_feature)
            #pdb.set_trace()

            # cls_list_dn = cls_list_dn.cuda()
            # reg_list_dn = reg_list_dn.cuda()
            # cls_list_up = cls_list_up.cuda()
            # reg_list_up = reg_list_up.cuda()
            
            prd_res = [] # initialize the prediction list
            for frame_num in range(0, 5):# (0, 5)
                
                cls_list_dn = cls_list_dn_raw[frame_num].detach().cpu().numpy()
                reg_list_dn = reg_list_dn_raw[frame_num].detach().cpu().numpy()
                cls_list_up = cls_list_up_raw[frame_num].detach().cpu().numpy()
                reg_list_up = reg_list_up_raw[frame_num].detach().cpu().numpy()

                
                cls_list = np.sqrt(cls_list_dn[0] * cls_list_up[0])
                cls_list[200] = cls_list[200] * np.sqrt(reg_list_dn[0, 2] * reg_list_up[0, 2]) 
                reg_list = reg_list_dn[0]
                reg_list[0:2] = np.array([-1, 1])[:, np.newaxis] * reg_list[0:2] + reg_list_up[0, 0:2]
                
            # 201 
            # argmax 
                '''Find background and foreground'''
                # print(idx, frame_num, "find background and foreground")
                cls_each = []
                idx_each = []

                feature = cls_list
                    
                feature_arr = np.array(feature[200, :])
                limt = feature_arr[np.argmax(feature_arr)]  # 阈值设置

                fgd_list = (feature[0:200, feature[200, :]>0.005]) 
                cls_each.extend(np.argmax(np.array(fgd_list), axis = 0))
                idx_each.extend([i for i in range(len(feature[200, :])) if feature[200, i]>0.005])

                # pdb.set_trace()
                
            # nms iou 
                # print(idx, frame_num, "num")
                origin_map = get_origin_map() 
                origin_map = origin_map[frame_num]
                proposals = []
                tot = 0
                for i in idx_each:
                    if reg_list[0, i] < reg_list[1, i] < 1:
                        xmin = origin_map[i] + reg_list[0, i]
                        xmax = origin_map[i] + reg_list[1, i]

                        xmin_score = xmax_score = 1

                        cls_score = reg_score =1
                        score = reg_list[2, i]
                        cls_idx = cls_each[tot]

                        proposals.append([xmin, xmax, xmin_score, xmax_score, cls_score, reg_score, score, cls_idx])
                    tot = tot + 1
                
                #pdb.set_trace()
            
                alpha = 0.5
                t1 = 0.001 # lower threshold
                t2 = 0.002 # higher threshold
                # proposals = torch.from_numpy(proposals)
                columns = ["xmin", "xmax", "xmin_score", "xmax_score", "cls_score", "reg_score", "score" ,"cls_idx"]
                proposals = pd.DataFrame(proposals, columns=columns)
                nms_res = soft_nms_proposal(proposals, alpha, t1, t2)
                # columns = ["xmin", "xmax", "xmin_score", "xmax_score", "cls_score", "reg_score", "score"]
                # df = pd.DataFrame(proposals, columns=columns)
                # df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)
                # pdb.set_trace()

                """Save information to prd_res_"""
                for i in range(0, len(nms_res['xmin'])):
                    prd_res.append([video_name[0], nms_res['xmin'][i], nms_res['xmax'][i], nms_res['score'][i], int(nms_res['cls_idx'][i])])
                    # pdb.set_trace()
            
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


            






    
