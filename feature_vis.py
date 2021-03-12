# coding: utf-8
import os
import json
import pdb
import random
from utils.dataset_utils import get_dataloader
from model.OsalModel import OsalModel

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import pdb

if __name__=='__main__':
    config_path = './config/config.json'
    try:
        f = open(config_path)
        config = json.load(f)
    except IOError:
        print('Model Building Error: errors occur when loading config file from ' + config_path)
        raise IOError

    dataLoader = get_dataloader(config, 'testing', 1, shuffle=False)
    num = 0
    all_features = []
    all_labels = []
    for features, cls_list, duration_list, video_name_list in dataLoader:
        num+=1
        print(cls_list, duration_list, video_name_list)
        if num==5:
            break
        if video_name_list[0] != "v_-01K1HxqPB8":
            # pdb.set_trace()
            continue
        all_features.append(features[0])
        labels = np.zeros((features.shape[2]))
        for label, duration in zip(cls_list[0], duration_list[0]):
            start_idx = int(duration[0] * 100)
            end_idx = int(duration[1] * 100)
            labels[start_idx:end_idx] = label
        all_labels.append(labels)
    pdb.set_trace()
    all_labels = np.concatenate(all_labels, axis=-1)
    all_features = np.concatenate(all_features, axis=-1).transpose()

    tsne = TSNE(n_components=3)
    result = tsne.fit_transform(all_features)
    result = (result-result.mean()) / result.std()

    result_dict = {}
    for i, point in enumerate(result):
        if result_dict.get(all_labels[i]) is None:
            result_dict[all_labels[i]] = []
        
        current_item = result_dict.get(all_labels[i])
        current_item.append(point)

    ax = plt.axes(projection='3d')
    legend = []
    colors = ['red', 'blue', 'y', 'purple', 'green']
    for idx, (label, points) in enumerate(result_dict.items()):
        # pdb.set_trace()
        point = np.stack(points)
        ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], color=colors[idx])

    plt.legend(list(result_dict.keys()))
    plt.savefig('res3D.jpg')
    # plt.show()

    # # pdb.set_trace()
        