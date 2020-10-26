import torch
import numpy as np 
import json, pdb
import torch

class LossCalculator():
    def __init__(self, config, device):
        self.perceptive_fields = np.array(config['perceptive_fields']) 
        self.perceptive_fields = self.perceptive_fields/100.
        self.feature_lens = config['feature_lens']
        self.origin_map = self.get_origin_map()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean', )
        self.bce_loss = torch.nn.BCELoss(reduction='mean')
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.nll_loss = torch.nn.NLLLoss()
        self.device = device

    def calc_loss(
        self, cls_list, reg_list, cls_list_final, reg_list_final, 
        cls_gt_, duration_list, epoch_num
    ):
        reg_gt_list, positive_indices = self.get_reg_gt(duration_list)

        cls_gt_list = cls_gt_
        for i in range(len(cls_gt_list)):
            cls_gt_list[i] = cls_gt_list[i].to(self.device)
        reg_ds_loss = torch.FloatTensor([0.]).to(self.device)
        cls_ds_loss = torch.FloatTensor([0.]).to(self.device)
        centerness_ds_loss = torch.FloatTensor([0.]).to(self.device)
        reg_us_loss = torch.FloatTensor([0.]).to(self.device)
        cls_us_loss = torch.FloatTensor([0.]).to(self.device)
        centerness_us_loss = torch.FloatTensor([0.]).to(self.device)
        
        # pdb.set_trace()
        for i, indice in enumerate(positive_indices):
            layer_idx, batch_idx, frame_idx = indice
            # pdb.set_trace()
            reg_result_ds = reg_list[layer_idx][batch_idx, :2, frame_idx]
            centerness_ds = reg_list[layer_idx][batch_idx, 2, frame_idx]
            cls_result_ds = cls_list[layer_idx][batch_idx, :200, frame_idx]
            
            reg_result_us = reg_list_final[layer_idx][batch_idx, :2, frame_idx]
            centerness_us = reg_list_final[layer_idx][batch_idx, 2, frame_idx]
            cls_result_us = cls_list_final[layer_idx][batch_idx, :200, frame_idx]

            # pdb.set_trace()
            cls_gt = cls_gt_list[layer_idx][batch_idx, 0, frame_idx]
            reg_gt = reg_gt_list[layer_idx][batch_idx, :2, frame_idx]
            reg_us_gt = (reg_result_ds - reg_gt).detach()
            centerness_gt = reg_gt_list[layer_idx][batch_idx, 2, frame_idx]

            # pdb.set_trace()
            reg_ds_loss += self.mse_loss(reg_result_ds, reg_gt)
            # pdb.set_trace()
            cls_ds_loss += self.ce_loss(cls_result_ds.unsqueeze(0), cls_gt.unsqueeze(0).long())
            # cls_ds_loss += self.nll_loss(log_cls_result_ds.unsqueeze(0), cls_gt.unsqueeze(0).long())
            centerness_ds_loss += self.mse_loss(centerness_ds, centerness_gt)
            reg_us_loss += self.mse_loss(reg_result_us, reg_us_gt)
            cls_us_loss += self.ce_loss(cls_result_us.unsqueeze(0), cls_gt.unsqueeze(0).long())
            # cls_us_loss += self.nll_loss(log_cls_result_us.unsqueeze(0), cls_gt.unsqueeze(0).long().detach())
            centerness_us_loss += self.mse_loss(centerness_us, centerness_gt)
        # pdb.set_trace()

        # calculate loss for background prediction
        bg_loss = torch.FloatTensor([0.]).to(self.device)
        for idx in range(len(self.feature_lens)):
            bg_ds_result = cls_list[idx][:, 200, :]
            bg_us_result = cls_list_final[idx][:, 200, :]
            bg_gt = cls_gt_list[idx][:, 1, :]
            bg_loss += self.bce_loss(bg_ds_result, bg_gt) + self.bce_loss(bg_us_result, bg_gt)

        num_pos_indice = len(positive_indices)
        # loss = (reg_ds_loss + reg_us_loss)/num_pos_indice*10
        # if epoch_num > 10:
        if epoch_num < 10:
            # print('all losses added')
            loss = (10*reg_ds_loss + 0.1*cls_ds_loss + centerness_ds_loss)/(1+num_pos_indice)*10 + bg_loss

        else:
            loss = (10*reg_ds_loss + 0.1*cls_ds_loss + centerness_ds_loss + 10*reg_us_loss + 0.1*cls_us_loss + centerness_us_loss)/(1+num_pos_indice)*10 + bg_loss

        # print(reg_ds_loss.item()/(1+num_pos_indice)*10, cls_ds_loss.item()/(1+num_pos_indice)*10, centerness_ds_loss.item()/(1+num_pos_indice)*10)
        # return loss , cls_us_loss/num_pos_indice
        return loss, bg_loss
        

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
        batch_size = len(duration_list)
        # pdb.set_trace()
        for video_idx, duration_list_per_video in enumerate(duration_list):
            reg_gt_list = []
            # initialize the list 
            for length in self.feature_lens:
                reg_gt_list.append(np.zeros((3, length)))

            # durations in video
            for duration in duration_list_per_video:
                start_time, end_time = duration
                layer_idx = self.allocate_layer(start_time, end_time)

                start_idx = int(start_time * 100) 
                end_idx = int(end_time * 100)
                start_idx = start_idx//2**(layer_idx+1) + start_idx//2**(layer_idx)%2
                end_idx = end_idx//2**(layer_idx+1) + end_idx//2**(layer_idx)%2
                start = self.origin_map[layer_idx][start_idx:end_idx+1]-start_time
                reg_gt_list[layer_idx][0, start_idx:end_idx+1] = start
                end = end_time - self.origin_map[layer_idx][start_idx:end_idx+1]
                reg_gt_list[layer_idx][1, start_idx:end_idx+1] = end
                reg_gt_list[layer_idx][2, start_idx:end_idx+1] = np.where(start>end, end, start)/np.where(start>end, start, end)
                # generate positive example indeices
                # pdb.set_trace()
                for len_idx in range(start_idx, end_idx):
                    positive_indices.append([layer_idx, video_idx, len_idx])
            # pdb.set_trace()
            reg_gt_list_all.append(reg_gt_list)
        reg_gt_list_final = []
        # pdb.set_trace()
        for layer_idx in range(len(self.feature_lens)):
            gt = np.stack([reg_gt[layer_idx] for reg_gt in reg_gt_list_all])
            gt = torch.Tensor(gt).to(self.device)
            reg_gt_list_final.append(gt)
        return reg_gt_list_final, positive_indices


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
    