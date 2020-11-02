import torch
import numpy as np 
import json, pdb
import torch, random
import torch.nn as nn
import torch.nn.functional as F
from utils.IOULoss import IOULoss

INF = 100000000

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        # pdb.set_trace()
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class LossCalculator():
    def __init__(self, config, device):
        self.perceptive_fields = np.array(config['perceptive_fields']) 
        self.perceptive_fields = self.perceptive_fields/200.
        self.feature_lens = config['feature_lens']
        self.layer_weight = config['layer_weight']
        self.origin_map = self.get_origin_map()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='sum', )
        self.bce_loss = torch.nn.BCELoss(reduction='mean')
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.nll_loss = torch.nn.NLLLoss()
        self.focal_loss = FocalLoss()
        self.device = device
        self.index_map = self.get_index_map()

    def calc_loss(
        self,cls_list_final, reg_list_final, 
        cls_gt_, duration_list, train_us, epoch_num
    ):
        reg_gt_list, positive_indices = self.get_reg_gt(duration_list)
        cls_gt_list = cls_gt_
        for i in range(len(cls_gt_list)):
            cls_gt_list[i] = cls_gt_list[i].to(self.device)

        reg_us_loss = torch.FloatTensor([0.]).to(self.device)
        cls_us_loss = torch.FloatTensor([0.]).to(self.device)
        centerness_us_loss = torch.FloatTensor([0.]).to(self.device)
        
        num_pos_indice = 0
        # pdb.set_trace()
        # 每个正样本分别计算loss，regressioni 和 centerness
        for i, indice in enumerate(positive_indices):
            layer_idx, batch_idx, frame_idx = indice

            num_pos_indice+=1            
            reg_result_us = reg_list_final[layer_idx][batch_idx, :2, frame_idx]
            centerness_us = reg_list_final[layer_idx][batch_idx, 2, frame_idx]

            reg_gt = reg_gt_list[layer_idx][batch_idx, :2, frame_idx]
            centerness_gt = reg_gt_list[layer_idx][batch_idx, 2, frame_idx]

            # pdb.set_trace()

            intersect = torch.min(reg_result_us, reg_gt).sum()
            union = torch.max(reg_result_us, reg_gt).sum() + 1e-7

            ious = (intersect + 1.0) / (union + 1.0)
            reg_us_loss += -torch.log(1e-6+ious)
            # gious = ious - (ac_uion - area_union) / ac_uion

            # reg_us_loss += self.mse_loss(reg_result_us, reg_gt) * centerness_gt # 用centerness作为权重
            centerness_us_loss += self.mse_loss(centerness_us, centerness_gt)

        # calculate classification loss
        for layer_idx in range(5):
            # pdb.set_trace()
            cls_us_loss += self.focal_loss(cls_list_final[layer_idx], cls_gt_list[layer_idx])

        loss = (10*reg_us_loss + centerness_us_loss)/(1+num_pos_indice)*10  + cls_us_loss*500
        return loss, (100*reg_us_loss/num_pos_indice, cls_us_loss*500, 10*centerness_us_loss/num_pos_indice)

    def get_reg_gt(self, duration_list):
        '''
        This kinds batch-size videos. 
        param:  
        :duration_list: list(list(tuple))
        return:
        :reg_gt_list: list(Tensor)
        :positive_indices: list(list(layer_idx, batch_idx, len_idx))
        '''
        # Firstly generate all the regression ground truth we need, then stack them together
        reg_gt_list_all = []
        positive_indices = []

        for video_idx, duration_list_per_video in enumerate(duration_list):
            reg_gt_list = []    # regression gt per video
            # initialize the list 
            for length in self.feature_lens:
                reg_gt_list.append(np.zeros((3, length)))

            # durations in video
            for duration in duration_list_per_video:
                start_time, end_time = duration
                for layer_idx in range(5):
                    start = self.origin_map[layer_idx]-start_time
                    end = end_time-self.origin_map[layer_idx]
                    start_end_tuple = np.stack([start, end], -1)
                    is_in_box = start_end_tuple.min(-1)>0
                    is_layer = \
                        (start_end_tuple.max(-1)>self.perceptive_fields[layer_idx]) & \
                        (start_end_tuple.max(-1)<self.perceptive_fields[layer_idx+1])
                    is_positive = np.logical_and(is_in_box, is_layer)
                    positive_indice = is_positive.nonzero()[0]

                    # generate positive example indeices
                    for len_idx in positive_indice:
                        positive_indices.append([layer_idx, video_idx, len_idx])
                        reg_gt_list[layer_idx][0, len_idx] = start[len_idx] # start
                        reg_gt_list[layer_idx][1, len_idx] = end[len_idx]   # end
                        reg_gt_list[layer_idx][2, len_idx] = min(start[len_idx], end[len_idx]) / max(start[len_idx], end[len_idx])
            reg_gt_list_all.append(reg_gt_list)
        reg_gt_list_final = []
        # pdb.set_trace()
        # concatnate to List[Tensor]
        for layer_idx in range(len(self.feature_lens)):
            gt = np.stack([reg_gt[layer_idx] for reg_gt in reg_gt_list_all])
            gt = torch.Tensor(gt).to(self.device)
            reg_gt_list_final.append(gt)
        return reg_gt_list_final, positive_indices

    def get_origin_map(self):
        origin_map = []
        for feature_len in self.feature_lens:
            mapping = np.arange(feature_len)
            time_gap = 100//feature_len
            mapping = mapping*time_gap + time_gap//2
            origin_map.append(mapping/100.)
        return origin_map

    def get_index_map(self):
        index_map = []
        divides = [50, 25, 13, 7, 4]
        origin_index = np.arange(100)
        for i in range(5):
            index_in_layer = np.floor(origin_index / (100./divides[i])) 
            index_map.append(index_in_layer.astype(np.int))
        # pdb.set_trace()
        return index_map



if __name__ == "__main__":
    from dataset_utils import get_dataloader
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
    