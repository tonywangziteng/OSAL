import torch
from torch.utils.data import Dataset, DataLoader
import pandas
import os
import os.path as osp
import numpy as np 
import json
import tqdm
import pdb
import h5py


# dataset: /media/e813/D/wzt/datasets/Activitynet/ 

# duration_second, duration_frame, annotation:[{segment, label}], feature_frame

class OsalDataset(Dataset):
    def __init__(
        self, 
        cfg, 
        mode='training', 
        dataset_name = "activityNet"
    ):
        assert dataset_name in ["activityNet", "thumos"], "Wrong dataset name, please check settings"
        if dataset_name == 'activityNet':
            self.__init_activitinet(cfg, mode)
        else:
            self.__init_thumos(cfg, mode)
        
    def __init_activitinet(self, cfg, mode):
        data_dir = cfg['data_dir']
        anno_path = cfg['anno_path'] 
        video_info_path = cfg['video_info_path'] 
        action_name_path = cfg['action_name_path']
        self.perceptive_fields = np.array(cfg['perceptive_fields'])/200. 
        self.feature_len = cfg['feature_lens']

        assert mode in ['training', 'validation', 'testing'], 'the mode should be training, validation or testing, instead of {}'.format(mode)
        self.mode = mode
        self.data_dir = data_dir
        try:
            self.data_file_list = os.listdir(self.data_dir) # files in dataset directory
        except BaseException:
            print('Errors occur when loading dataset')
        
        # load dataset information
        all_info = pandas.read_csv(video_info_path)
        if self.mode == 'training':
            self.data_info = all_info[all_info.subset == 'training']
        elif self.mode == 'validation' or 'testing':
            self.data_info = all_info[all_info.subset == 'validation']

        self.video_name_list = self.data_info.video.tolist()

        # load annotations 
        try:
            anno_file = open(anno_path)
        except IOError:
            print('Errors occur when loading annotations from {}'.format(anno_path))
        else:
            self.annotations = json.load(anno_file)

        # load action names
        action_name = pandas.read_csv(action_name_path)
        self.action_name = action_name['action'].tolist()
        self.index_map = self.get_index_map()
        self.origin_map = self.get_origin_map()

    def calc_gt(self, video_name:str):
        """
        classification ground truth should be between batch_size * 100 * (200+1) 
        """
        video_info = self.annotations[video_name]
        video_anno = video_info['annotations']
        # pdb.set_trace()

        # calculate the basic length information about the video
        video_real_frame = video_info['duration_frame']
        video_real_second = video_info['duration_second']
        video_feature_frame = video_info['feature_frame']
        video_feature_second = float(video_feature_frame) / video_real_frame * video_real_second

        # initialize cls_list
        cls_gt = []
        boundary_list = []
        cls_list = []
        start_end_score = np.zeros((2, 100))    # start end 得分
        for length in self.feature_len:
            # cls_gt.append(np.zeros((2, length))) # 多分类
            cls_gt.append(np.zeros((201, length)))

        for anno in video_anno:
            action_name = anno['label']
            # print(action_name)
            name_index = self.action_name.index(action_name)
            start_time = max((min(1, anno['segment'][0]/video_feature_second)), 0)
            end_time = max((min(1, anno['segment'][1]/video_feature_second)), 0)

            # get the layer where the ground truth belongs to
            anno_layer_index = 0
            for i in range(5):
                start = self.origin_map[i]-start_time
                end = end_time - self.origin_map[i]
                start_end = np.stack([start, end], -1)
                is_above_layer = \
                    (start_end.max(-1)>self.perceptive_fields[i]) & \
                    (start_end.max(-1)<self.perceptive_fields[i+1]) & \
                    (start_end.min(-1)>0)
                    
                if is_above_layer.any():
                    anno_layer_index = i

            # background and classification ground truth
            for i in range(len(self.index_map)):
                if i > anno_layer_index:
                    continue

                cls_gt[i][name_index, np.logical_and(self.origin_map[i]>start_time, self.origin_map[i]<end_time)] = 1
                cls_gt[i][200, np.logical_and(self.origin_map[i]>start_time, self.origin_map[i]<end_time)] = 1
            # pdb.set_trace()
            # start and end score
            std = (end_time - start_time) / 20. + 1e-6
            amplitude_map = 1/(np.sqrt(2*np.pi)*std)
            start_end_score[0] += self.gaussian_pdf(start_time, std) / amplitude_map
            start_end_score[1] += self.gaussian_pdf(end_time, std) / amplitude_map

            boundary_list.append((start_time, end_time))
            cls_list.append(name_index)
        
        return cls_gt, boundary_list, cls_list, start_end_score

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

    def calc_iou(self, start, end, gt_start, gt_end):
        start_min = np.where(start<gt_start, start, gt_start)
        start_max = np.where(start>gt_start, start, gt_start)
        end_min = np.where(end<gt_end, end, gt_end)
        end_max = np.where(end>gt_end, end, gt_end)
        intersection = end_min-start_max
        union = end_max - start_min
        return np.clip(intersection*1.0/union, 0, 1)

    def get_index_map(self):
        index_map = []
        divides = [50, 25, 13, 7, 4]
        origin_index = np.arange(100)
        for i in range(5):
            index_in_layer = np.floor(origin_index / (100./divides[i])) 
            index_map.append(index_in_layer.astype(np.int))
        # pdb.set_trace()
        return index_map

    def gaussian_pdf(self, mean, std):
        x = (np.arange(100)+0.5)/100.
        std = std+1e-6
        y = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
        return y

    def get_origin_map(self):
        origin_map = []
        for feature_len in self.feature_len:
            mapping = np.arange(feature_len)
            time_gap = 100//feature_len
            mapping = mapping*time_gap + time_gap//2
            origin_map.append(mapping/100.)
        return origin_map

    def __getitem__(self, index):
        video_name = self.video_name_list[index]
        # print(video_name)
        feature = pandas.read_csv(osp.join(self.data_dir, video_name+'.csv'))
        feature = feature.values   
        
        # calculate ground truth
        cls_gt, boundary_list, cls_list, start_end_score = self.calc_gt(video_name)
        # pdb.set_trace()

        # feature: batch_size * len(100) * feature_depth(400)

        if self.mode == 'testing':
            return feature, cls_gt, boundary_list, video_name, cls_list
        else:
            return feature, cls_gt, boundary_list, video_name, start_end_score

    def __len__(self):
        return len(self.video_name_list) 

def collate_function(batch):
    feature_list, cls_gt_list, duration_list = [], [], []
    video_name_list, cls_list = [], []
    start_end_list = []
    # 如果最后一个是list， 说明mode=='testing'
    if isinstance(batch[0][-1], list):
        mode = 'testing'
    else:
        mode = 'training'
    # batch里面有batch_size个list
    for idx, element in enumerate(batch):
        feature_list.append(torch.Tensor(element[0]))
        # concat cls_gt
        for cls_idx, cls_gt in enumerate(element[1]):
            if idx == 0:
                cls_gt_list.append([torch.Tensor(cls_gt)])
            else:
                cls_gt_list[cls_idx].append(torch.Tensor(cls_gt))
        duration_list.append(element[2])
        video_name_list.append(element[3])
        if mode == 'training':
            start_end_list.append(torch.Tensor(element[4]))
        else:
            cls_list.append(element[4])
    features = torch.stack(feature_list, 0)
    features = features.permute(0, 2, 1) # conv1 reaquires shape of (bs*channels*length)
    cls_gt = []
    for cls_gt_stacked in cls_gt_list:
        cls_gt.append(torch.stack(cls_gt_stacked, 0))
    if mode=='testing':
        return features, cls_list, duration_list, video_name_list
    else:
        start_end = torch.stack(start_end_list, 0)
        return features, cls_gt, duration_list, video_name_list, start_end


def get_dataloader(cfg, mode, batch_size, shuffle = True, num_worker = 4):
    r"""
    returns:
    :feature: (Tensor) batch_size*400*100
    :cls_gt: (List(Tensor)) [batch_size*2*length] 
    :duration_list: (List(List(Tuple(start, end)))) 
    """
    
    dataset = OsalDataset(
        cfg=cfg, 
        mode=mode, 
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_function)
    return data_loader

        
if __name__ == '__main__':
    f = h5py.File('../THUMOS14_feat/thumos14_i3d_features_flow_with_ucf101.hdf5', 'r')
    pdb.set_trace()
    try:
        f = open("/media/e813/D/wzt/codes/wzt_OSAL/config/config.json")
        overall_config = json.load(f)
    except IOError:
        # print('Model Building Error: errors occur when loading config file from '+config_path)
        raise BaseException
    train_loader = get_dataloader(overall_config, 'training', 2)
    for idx, (raw_feature, cls_gt, duration_list) in enumerate(train_loader):
        print(cls_gt)
        # print(raw_feature)
        pdb.set_trace()

