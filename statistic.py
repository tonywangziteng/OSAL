import torch
from torch.utils.data import Dataset, DataLoader
import pandas
import os
import os.path as osp
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import math


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
            anno_path: str, video_info_path: str,
            mode='training'
    ):
        assert mode in ['training', 'validation',
                        'testing'], 'the mode should be training, validation or testing, instead of {}'.format(mode)
        self.mode = mode

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
        #action_name = pandas.read_csv(action_name_path)
        #self.action_name = action_name['action'].tolist()

    def calc_gt(self, video_name: str):
        """
        classification ground truth should be between batch_size * 100 * (200+1)
        """
        video_info = self.annotations[video_name]
        video_anno = video_info['annotations']

        # calculate the basic length information about the video
        video_real_frame = video_info['duration_frame']
        video_real_second = video_info['duration_second']
        video_feature_frame = video_info['feature_frame']
        video_feature_second = float(video_feature_frame) / video_real_frame * video_real_second

        cls_gt = np.zeros((100, 201))  # first 200 dims are classes, the last one is background dim
        boundary_list = []
        for anno in video_anno:
            action_name = anno['label']
            #name_index = self.action_name.index(action_name)
            start_time = max((min(1, anno['segment'][0] / video_feature_second)), 0)
            #start_idx = int(start_time * 100)
            end_time = max((min(1, anno['segment'][1] / video_feature_second)), 0)
            #end_idx = int(end_time * 100)

            #cls_gt[start_idx:end_idx, name_index] = 1
            #cls_gt[start_idx:end_idx, 200] = 1
            boundary_list.append((end_time - start_time) * 100)

        #return cls_gt, boundary_list
        return boundary_list

if __name__ == '__main__':
    train_dataset = OsalDataset(
        anno_path='/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/anet_anno_action.json',
        video_info_path="/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/video_info_new.csv"
    )
    csv_path = "/media/e813/D/wzt/codes/Pytorch-BMN/data/activitynet_annotations/video_info_new.csv"
    csvfile = pandas.read_csv(csv_path)
    vname = list(csvfile["video"])
    ans_list = []
    for v_name in vname:
        csv_list = train_dataset.calc_gt(v_name)
        if len(csv_list) != 0:
            ans_list.extend(csv_list)

    print(ans_list)
    plt.figure()
    x = np.arange(11)
    y = np.zeros(11)
    for i in ans_list:
        y[math.floor(i/10)] = y[math.floor(i/10)]+1
    df = pandas.DataFrame({"x-axis": x, "y-axis": y})
    sns.barplot("x-axis", "y-axis", palette="RdBu_r", data=df)
    plt.xticks(rotation=90)
    plt.savefig('fig2.png')





