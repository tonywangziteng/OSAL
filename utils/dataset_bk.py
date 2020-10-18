import torch
from torch.utils.data import Dataset, DataLoader
import pandas
import os
import os.path as osp
import numpy as np 
import json
import tqdm
import pdb
# dataset: /media/e813/D/wzt/datasets/Activitynet/ 

# duration_second, duration_frame, annotation:[{segment, label}], feature_frame

class OsalDataset(Dataset):
    r"""  
    Arguments
        :data_dir: full path of the dataset 
        :anno_path: full path of the annotation file
        :mode: 'training', 'validation', 'testing' | decide what the dataset is used for

    create the dataset for train, evaluation, and test.
    mainly aims at calculating ground truth target
    """
    def __init__(
        self, 
        data_dir:str, anno_path:str, video_info_path:str, action_name_path:str, 
        mode='training'
    ):
        assert mode in ['training', 'validation', 'testing'], 'the mode should be training, validation or testing, instead of {}'.format(mode)
        self.mode = mode
        self.data_dir = data_dir
        try:
            self.data_file_list = os.listdir(self.data_dir) # files in dataset directory
        except BaseException:
            print('Errors occur when loading dataset')
        
        # load dataset information
        all_info = pandas.read_csv(video_info_path)
        self.data_info = all_info[all_info.subset == self.mode]
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

    def calc_gt(self, video_name:str):
        """
        classification ground truth should be between batch_size * 100 * (200+1) 
        """
        video_info = self.annotations[video_name]
        video_anno = video_info['annotations']
        pdb.set_trace()

        # calculate the basic length information about the video
        video_real_frame = video_info['duration_frame']
        video_real_second = video_info['duration_second']
        video_feature_frame = video_info['feature_frame']
        video_feature_second = float(video_feature_frame) / video_real_frame * video_real_second

        cls_gt = np.zeros((100, 201)) # first 200 dims are classes, the last one is background dim
        boundary_list = []
        for anno in video_anno:
            action_name = anno['label']
            name_index = self.action_name.index(action_name)
            start_time = max((min(1, anno['segment'][0]/video_feature_second)), 0)
            start_idx = int(start_time * 100) 
            end_time = max((min(1, anno['segment'][1]/video_feature_second)), 0)
            end_idx = int(end_time * 100)

            cls_gt[start_idx:end_idx, name_index] = 1
            cls_gt[start_idx:end_idx, 200] = 1
            boundary_list.append((start_time, end_time))
        
        return cls_gt, boundary_list

    def __getitem__(self, index):
        video_name = self.video_name_list[index]
        feature = pandas.read_csv(osp.join(self.data_dir, video_name+'.csv'))
        feature = feature.values   
        
        # calculate ground truth
        cls_gt, boundary_list = self.calc_gt(video_name)

        # feature: batch_size * len(100) * feature_depth(400)
        return feature, cls_gt, boundary_list

    def __len__(self):
        return len(self.video_name_list) 

def collate_function(batch):
    feature_list, cls_gt_list, duration_list = [], [], []
    for idx, element in enumerate(batch):
        feature_list.append(torch.Tensor(element[0]))
        cls_gt_list.append(torch.Tensor(element[1]))
        duration_list.append(element[2])
    features = torch.stack(feature_list, 0)
    features = features.permute(0, 2, 1) # conv1 reaquires shape of (bs*channels*length)
    cls_gt = torch.stack(cls_gt_list, 0)
    # pdb.set_trace()
    return features, cls_gt, duration_list

def get_dataloader(mode, batch_size):
    """
    returns:
    :feature: (Tensor) batch_size*100*400
    :cls_gt: (Tensor) batch_size*100*201
    :duration_list: (List(List(Tuple(start, end)))) 
    """
    dataset = OsalDataset(
        data_dir='/media/e813/D/wzt/datasets/Activitynet/', 
        anno_path='/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/anet_anno_action.json', 
        video_info_path="/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/video_info_new.csv", 
        action_name_path="/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/action_name.csv", 
        mode=mode
    )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_function)
    return data_loader

        
if __name__ == '__main__':
    train_dataset = OsalDataset(
        data_dir='/media/e813/D/wzt/datasets/Activitynet/', 
        anno_path='/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/anet_anno_action.json', 
        video_info_path="/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/video_info_new.csv", 
        action_name_path="/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/action_name.csv"
    )
    train_loader = get_dataloader('training')
    for idx, (raw_feature, cls_gt, duration_list) in enumerate(train_loader):
        print(cls_gt)
        # print(raw_feature)
        pdb.set_trace()

