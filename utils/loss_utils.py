import torch
import numpy as np 
import json, pdb
from dataset_utils import get_dataloader
import torch

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
        reg_gt = self.

    def get_reg_gt(self, duration_list):
        '''
        This kinds batch-size videos. 
        param:  
        :duration_list: list(list(tuple))
        '''
        # Firstly generate all the regression ground truth we need, then stack them together
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


class OsalLoss():
    def __init__(self):
        self.reg_loss_fn = torch.nn.MSELoss()
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.epoch_train_loss = 0
        self.epoch_cls_loss = 0
        self.epoch_reg_loss = 0

    def LossCalc(self, cls_list, cls_list_final, reg_list, reg_list_final, cls_gt, duration_list, clip_len_list,
                 num_anno, layer_index_list):
        mask_list = []
        num_in_layers = []
        num_sample = 0.
        for i in range(5):
            mask_list.append(cls_gt[i][:, 200])
            num_in_layers.append(torch.sum(
                torch.where(mask_list[i] > 0.5, torch.ones(clip_len_list[i], dtype=torch.float32),
                            torch.zeros(clip_len_list[i], dtype=torch.float32))))
        for id, i in enumerate(pow(2, np.array([0, 1, 2, 3, 4]))):
            num_sample += num_in_layers[id] * i
        neg_sample_ratio = max(num_sample / (50 * 5), 0.)
        for clip_idx in range(num_anno):
            layer_idx = layer_index_list[clip_idx]
            a = torch.Tensor(duration_list[clip_idx])
            a = a.expand((clip_len_list[layer_idx], 2))
            reg_gt = a.transpose(0, 1)
            self.epoch_reg_loss += self.reg_loss_fn(reg_list[layer_idx].transpose(0, 1) * mask_list[layer_idx],
                                                    mask_list[layer_idx] * reg_gt)  # 粗reg loss

            reg_seq = torch.tensor([i for i in range(clip_len_list[layer_idx])], dtype=torch.float32)
            reg_seq = reg_seq.expand((2, clip_len_list[layer_idx]))
            self.epoch_reg_loss += self.reg_loss_fn(
                mask_list[layer_idx] * (reg_list_final[layer_idx].transpose(0, 1) + reg_seq),
                mask_list[layer_idx] * reg_gt)  # 细reg loss

            self.epoch_cls_loss = self.epoch_cls_loss / 2. + self.cls_loss_fn(
                mask_list[layer_idx] * cls_list[layer_idx].transpose(0, 1),
                mask_list[layer_idx] * torch.Tensor(cls_gt[layer_idx]).transpose(0, 1)) + \
                                  self.cls_loss_fn(
                                      mask_list[layer_idx] * cls_list_final[layer_idx].transpose(0, 1),
                                      mask_list[layer_idx] * torch.Tensor(cls_gt[layer_idx]).transpose(0, 1))
            # 正样本classification loss
            for i in range(5):
                neg_mask = 1 - mask_list[i]
                self.epoch_cls_loss = self.epoch_cls_loss / 2. + neg_sample_ratio * self.cls_loss_fn(
                    neg_mask * cls_list[i].transpose(0, 1), neg_mask * cls_list[i].transpose(0, 1)) + \
                                      neg_sample_ratio * self.cls_loss_fn(
                                            neg_mask * cls_list_final[i].transpose(0, 1), neg_mask * cls_list[i].transpose(0, 1))
                # 负样本clsloss 乘2因为短视野数据量减半了，loss加回来
        return self.epoch_reg_loss, self.epoch_cls_loss


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
    