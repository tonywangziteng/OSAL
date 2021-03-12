# coding: utf-8
import json
import pdb
import numpy as np
import pandas
from utils.mAP_utils import get_preds_list, voc_eval, get_gt_dict_2
# from data import class_map_dict_inv, class_map_dict 

# some params
test_gt_json_path = './output/anet_anno_action.json'
test_preds_json_path = './output/result_proposal.json'
action_name_path = './output/action_name.csv'
video_info_path = "/home/e813/wzt_code/Pytorch-BMN/data/activitynet_annotations/video_info_new.csv"

nms_thres = 0.02
iou_thres = 0.5
score_thres = 0.02
class_nb = 200

# test_preds = json.load(open('./test_results_raw.json', 'r'))
# test_preds = parse_preds(test_preds, nms_thres, class_nb)
# gt_dict = get_gt_dict(test_gt_json_path, class_map_dict)
all_info = pandas.read_csv(video_info_path)
test_video_names = all_info[all_info.subset == 'validation'].video.values.tolist()    # ndarray

test_preds, test_name = get_preds_list(test_preds_json_path)
print("number of proposals: ", len(test_preds))
gt_dict = get_gt_dict_2(test_gt_json_path, action_name_path, test_video_names)

print('Test Configs:\nNMS threshold: {}\nIoU threshold: {}\n'.format(nms_thres, iou_thres))

ap_t = 0.
for ii in range(class_nb):
    # pdb.set_trace()
    recall, precision, ap = voc_eval(ii, gt_dict, test_preds, test_name, iou_thres, score_thres)
    ap_t += ap
    # print('TEST: Class {:<2} [{:<17}]: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}'.format(ii, class_map_dict_inv[ii], recall, precision, ap))
    print('TEST: Class {:<2} : AP: {:.4f}'.format(ii, ap))
mAP = ap_t / class_nb
print('test mAP: {:.4f}'.format(mAP))

# write_file
preds = {}
for i in test_preds:
    if i[0] not in preds:
        preds[i[0]] = []
    preds[i[0]].append([i[0], i[1], i[2], i[3], i[-1]])
# sort
for k, v in preds.items():
    # item = sorted(v, key=lambda x: x[3], reverse=True)
    item = sorted(v, key=lambda x: x[1], reverse=False)
    item = [i for i in item if i[3] > score_thres]
    preds[k] = item
f = open('processed_test_result.txt', 'w')
for k, v in preds.items():
    for i in v:
        line = '{} {} {} {} {}\n'.format(i[0], i[1], i[2], i[3], i[-1])
        f.write(line)
f.close()
