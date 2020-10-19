import torch
import numpy as np 
import json, pdb
from dataset_utils import get_dataloader

class LossCalculator():
    def __init__(self, config):
        self.perceptive_fields = np.array(config['perceptive_fields']) 
        self.perceptive_fields = self.perceptive_fields/100.
        self.feature_lens = config['feature_lens']
        self.origin_map = self.get_origin_map()

    def calc_loss(
        self, cls_list, reg_list, cls_list_final, reg_list_final, 
        cls_gt, duration_list
    ):
        # calculate classification loss
        pass

    def get_reg_gt(self, duration_list):
        '''
        This kinds batch-size videos. 
        param:  
        :duration_list: list(list(tuple))
        '''
        #TODO: Firstly generate all the regression ground truth we need, then stack them together
        reg_gt_list_all = []
        batch_size = len(duration_list)
        # pdb.set_trace()
        for video_idx, duration_list_per_video in enumerate(duration_list):
            reg_gt_list = []
            # initialize the list 
            for length in self.feature_lens:
                reg_gt_list.append(np.zeros((length, 3)))

            # durations in video
            for duration in duration_list_per_video:
                start_time, end_time = duration
                layer_idx = self.allocate_layer(start_time, end_time)
                
                start_idx = int(start_time * 100) 
                end_idx = int(end_time * 100)
                start_idx = start_idx//2**(layer_idx+1) + start_idx//2**(layer_idx)%2
                end_idx = end_idx//2**(layer_idx+1) + end_idx//2**(layer_idx)%2
                
                reg_gt_list[layer_idx][start_idx:end_idx, 0] = self.origin_map[layer_idx][start_idx:end_idx]-start_time
                reg_gt_list[layer_idx][start_idx:end_idx, 1] = end_time - self.origin_map[layer_idx][start_idx:end_idx]
            pdb.set_trace()
            reg_gt_list_all.append(reg_gt_list)
        reg_gt_list_final = []
        pdb.set_trace()
        for layer_idx in range(len(self.feature_lens)):
            gt = np.stack([reg_gt[layer_idx] for reg_gt in reg_gt_list_all])
            gt = torch.Tensor(gt)
            reg_gt_list_final.append(gt)
        return reg_gt_list_final


    def allocate_layer(self, start_time, end_time):
        """
        Get to know which layer the annotation belongs to 
        """
        duration = end_time - start_time
        i = 0
        for i in range(len(self.perceptive_fields)):
            if duration < self.perceptive_fields[i]:
                return i
        return i

    def get_origin_map(self):
        origin_map = []
        for feature_len in self.feature_lens:
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
    loss_calculator= LossCalculator(config)
    data_loader = get_dataloader(config, 'training', batch_size=2)
    mapping = loss_calculator.origin_map
    for idx, (raw_feature, cls_gt, duration_list) in enumerate(data_loader):
        reg_gt_coarse = loss_calculator.get_reg_gt(duration_list)
        pdb.set_trace()
    