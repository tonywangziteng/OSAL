import cv2
import os.path as osp
import pdb
import json
import pandas as pd
import torch
import numpy as np 
import torch.nn as nn
from model.OsalModel import OsalModel
from utils.nms_utils import soft_nms_proposal
import os


def get_origin_map():
    origin_map = []
    for feature_len in config['feature_lens']:
        mapping = np.arange(feature_len)
        time_gap = 100//feature_len
        mapping = mapping*time_gap + time_gap//2
        origin_map.append(mapping/100.)
    return origin_map


def find_action(feature:torch.Tensor) -> list:
    '''从模型中跑出来'''
    raw_feature = feature.cuda().permute(0,2,1)
    cls_list_dn_raw, reg_list_dn_raw, start_end  = model(raw_feature) 

    prd_res = []
    perceptive_bounds = []
    perceptive_fields = config["perceptive_fields"]
    feature_lens = config["feature_lens"]

    origin_map = get_origin_map() 
    for i in range(len(origin_map)):
        origin_map[i] = np.array(origin_map[i])
    origin_map_extended = np.concatenate(origin_map)

    perceptive_fields = np.array(perceptive_fields) / 200.
    for layer_index in range(len(cls_list_dn_raw)):
        feature_len = feature_lens[layer_index]
        pb = np.array([perceptive_fields[layer_index], perceptive_fields[layer_index+1]])
        pb_expanded = np.repeat(pb[np.newaxis, :], feature_len, 0)
        perceptive_bounds.append(pb_expanded)
    perceptive_bounds_expanded = np.concatenate(perceptive_bounds, 0)

    # 不同层concat到一起
    cls_expanded = torch.cat(cls_list_dn_raw, -1).squeeze(0).detach().cpu().numpy()
    reg_expanded = torch.cat(reg_list_dn_raw, -1).squeeze(0).detach().cpu().numpy()

    # Find background and foreground
    start_end = start_end[0].detach().cpu().numpy()
    cls_result = cls_expanded[:200] # 多分类
    fg_result = cls_expanded[200]# * reg_expanded[2] # 二分类 * centerness

    reg_result = reg_expanded[:2]

    start = origin_map_extended - reg_result[0]
    start = np.clip(start, 0, 1)
    end = origin_map_extended + reg_result[1]
    end = np.clip(end, 0, 1)

    start_idx_floor = np.floor(start*100).astype(np.int)
    start_idx_floor[start_idx_floor==100] = 99
    start_score = start_end[0, start_idx_floor]
    
    end_idx_floor = np.floor(end*100).astype(np.int)
    end_idx_floor[end_idx_floor==100] = 99
    end_score = start_end[1, end_idx_floor]

    fg_result = fg_result * np.sqrt(start_score * end_score)

    fg_pos = \
        (np.max(reg_result, axis=0) > perceptive_bounds_expanded[:, 0]) & \
        (np.max(reg_result, axis=0) < perceptive_bounds_expanded[:, 1]) & \
        (fg_result > config['bg_threshold'])
    fg_indice = fg_pos.nonzero()

    cls_positive = cls_result[:, fg_pos]
    cls_each = np.argmax(cls_positive, axis=0)
    idx_each = fg_indice[0]

    # 把所有的正样本放到proposals这个list里面
    proposals = []
    for i, i_pos in enumerate(idx_each):
        xmin = np.clip(origin_map_extended[i_pos]-reg_result[0, i_pos], 0, 1)
        xmax = np.clip(origin_map_extended[i_pos]+reg_result[1, i_pos], 0, 1)   #满足条件的时候把偏移量加上

        # score = reg_list[2, i]  #centerness
        # score = fg_result[i_pos] # foreground and background
        score = fg_result[i_pos]
        cls_idx = cls_each[i]

        proposals.append([xmin, xmax, score, cls_idx]) #根据index_each给每个位置打分
                    
    alpha = 0.2
    t1 = 0.0002 # lower threshold
    t2 = 0.1 # higher threshold
    columns = ["xmin", "xmax", "score" ,"cls_idx"]
    proposals = pd.DataFrame(proposals, columns=columns)
    nms_res = soft_nms_proposal(proposals, alpha, t1, t2)
    
    # pdb.set_trace()
    """Save information to prd_res_"""
    for i in range(len(nms_res['xmin'])):
        if nms_res['score'][i]>config['bg_threshold']:
            prd_res.append((nms_res['xmin'][i], nms_res['xmax'][i], nms_res['score'][i], action_name_list[int(nms_res['cls_idx'][i])]))

    return prd_res

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

config_path = './config/config.json'
try:
    f = open(config_path)
    config = json.load(f)
except:
    print('Model Building Error: errors occur when loading config file from ' + config_path)
    raise IOError

data_dir = config['data_dir']
anno_path = config['anno_path'] 
video_info_path = config['video_info_path'] 
action_name_path = config['action_name_path']

action_name = pd.read_csv(action_name_path)
action_name_list = action_name['action'].tolist()

def load_video(video_name):
    video_dir = "../ActivityNet/ActivityNetDataset"
    # video_name = "v_--1DO2V4K74" 
    video_full_name = video_name + ".mp4"
    video_path = osp.join(video_dir, video_full_name)
    capture = cv2.VideoCapture(video_path)
    return capture

# 加载测试视频的名字
all_info = pd.read_csv(video_info_path)
test_info = all_info[all_info.subset == 'validation']
video_name_list = test_info.video.tolist()
# pdb.set_trace()

anno_file = open(anno_path)
annotations = json.load(anno_file)



# TODO: 
def anal_pred_result(ratio:float, pred_result:list)->list:
    pred_res_list = []
    for start, end, score, label in pred_result:
        if ratio >= start and ratio<=end:
            pred_res_list.append(label)
    return pred_res_list

def anal_gt_result(ratio:float, gt_anno:list)->str:
    for anno in gt_anno:
        start, end = anno['segment']
        start = start/duration_second
        end = end/duration_second
        label = anno['label']
        if ratio>=start and ratio<=end:
            return label

if __name__ == '__main__':

    # out_stream = cv2.VideoWriter()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 初始化模型
    model = OsalModel()
    model = model.cuda()

    # 加载参数
    weight_dir = config['checkpoint_dir']
    weight_path = osp.join(weight_dir, 'ship_connection/epoch30_3.6463678769163184.pth.tar')
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for video_name in video_name_list[9:20]:

        capture = load_video(video_name)
        print('processing '+video_name)

        # 视频相关信息
        frame_cnt = annotations[video_name]['feature_frame']
        duration_second = annotations[video_name]['duration_second']
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(width, height)

        out_stream = cv2.VideoWriter('output_video/'+video_name+'.avi', fourcc, frame_rate, (int(width), int(height)))

        # 获取数据
        feature = pd.read_csv(osp.join(data_dir, video_name+'.csv'))
        feature = feature.values  
        feature = torch.Tensor(feature).unsqueeze(0)

        # 获取标注
        annotation = annotations[video_name]['annotations'] # [{'segment': , 'label': }]

        #infer
        prediction_result = find_action(feature)
        if len(prediction_result) == 0:
            print('no result')
            # raise ValueError
            continue

        cnt = 0
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            pred_action_list = anal_pred_result(cnt/frame_cnt, prediction_result)
            gt_result = anal_gt_result(cnt/frame_cnt, annotation)
            
            if len(pred_action_list)>0:
                pred_result = 'prediction: '
                for label in pred_action_list:
                    pred_result += label+'/'
                cv2.putText(frame, pred_result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # print(pred_action_list)
            if gt_result is not None:
                cv2.putText(frame, 'ground truth: '+gt_result, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # cv2.imshow('frame', frame)
            # k = cv2.waitKey(1)
            # if k == 27:
            #     break
            out_stream.write(frame)
            cnt+=1
        capture.release()
        out_stream.release()

    # pdb.set_trace()

