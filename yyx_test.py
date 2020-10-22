import json
import multiprocessing as mp
import os
import random
import pdb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset_utils import get_dataloader
from model.OsalModel import OsalModel

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

    #checkpoint = torch.load(opt.checkpoint_path + '9_param.pth.tar')
    #model.load_state_dict(checkpoint['satate_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #start_epoch = checkpoint['epoch']

    test_dataloader = get_dataloader(config, 'testing', batch_size=2)

    print("Process data and save.")
    with torch.no_grad():
        model.eval()
        for idx, (raw_feature, cls_gt, duration_list) in enumerate(test_dataloader):
            raw_feature = raw_feature.cuda()
            cls_list_down, reg_list_down, cls_list_up, reg_list_up = model(raw_feature)
            #pdb.set_trace()

            cls_list_down = cls_list_down[0].detach().cpu().numpy()
            reg_list_down = reg_list_down[0].detach().cpu().numpy()
            cls_list_up = cls_list_up[0].detach().cpu().numpy()
            reg_list_up = reg_list_up[0].detach().cpu().numpy()

            cls_list = cls_list_down * cls_list_up
            reg_list = reg_list_down + reg_list_up

            # 201 
            # argmax
            fgd_list = []
            '''Find background and foreground'''
            for feature in cls_list:
                fgd_list.append(feature[:, feature[200, :]>0.5])
                pdb.set_trace()
            
            

            # nms iou 

            # map

            






    
