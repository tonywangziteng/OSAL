# coding: utf-8

import random
import numpy as np
import os
import json
from datetime import datetime

# TODO: aumotacially compute ratio
# ratio=0.03
def sample_train(train_path, new_train_path, ratio=0.3):
	f1 = open(train_path, 'r')
	f2 = open(new_train_path, 'w')

	for line in f1:
		line_sep = line.strip().split(' ')
		action_class = line_sep[4]

		if action_class == 'background':
			if random.random() < ratio:
				f2.write(line)
		else:
			f2.write(line)

	f1.close()
	f2.close()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)

#### for mAP evaluating
def get_gt_dict(gt_path, class_map_dict):
    # key: video_name
    # value: [[start_sec, end_sec, class_idx], ...]
    gt_dict = {}
    f = json.load(open(gt_path, 'r'))
    for k in f:
        gt_dict[k] = []
        duration = f[k]['duration']
        content = f[k]
        gt_second_stamps = content['gt_second_stamps']
        action_names = content['action_name']
        for i in range(len(gt_second_stamps)):
            # remove the falsely annotated items
            if gt_second_stamps[i][1] > duration:
                continue
            cur_content = [gt_second_stamps[i][0], gt_second_stamps[i][1], class_map_dict[action_names[i]]]
            gt_dict[k].append(cur_content)
    return gt_dict


def nms_detection(items, overlap=0.4):
    # items: nested lists
    # [[video_name, new_start_time, new_end_time, score, fg_prob, class_prob, prop_score, class_idx], ...]
    if len(items) == 0:
        return items
    
    intervals = np.asarray([[item[1], item[2]] for item in items], np.float32)
    scores = np.asarray([item[3] for item in items], np.float32)

    t1 = intervals[:, 0]
    t2 = intervals[:, 1]
    area = (t2 - t1).astype(float)

    ind = np.argsort(scores)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]

    ret = []
    for i in pick:
        ret.append(items[i])

    return ret

def parse_preds(preds, nms_thres, class_nb):
    # preds: dict
    # key: video_name
    # values: [[video_name, new_start_time, new_end_time, score, fg_prob, class_prob, prop_score, class_idx], ...]
    pred_items = []
    for k, v in preds.items():
        for act in range(class_nb):
            items = [i for i in v if i[-1] == act]  # select the "act"(cls) items
            if len(items) <= 0:
                continue
            ret_items = nms_detection(items, overlap=nms_thres)
            pred_items.extend(ret_items)
    return pred_items

def get_preds_list(test_preds_json_path):
    # preds : [[video_name, start_time, end_time, score, class_idx], ...]
    f = json.load(open(test_preds_json_path, 'r'))
    preds = []
    for k, v in f.items():
        preds = [i for i in v]
    return preds


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
        
def voc_eval(classidx,
             gt_dict,
             test_preds,
             iou_thres=0.5,
             score_thres=0.01,
             use_07_metric=False):
    
    # gt_dict = { 'vid_name': [start, end, class index] }
    # test_preds = [video_name, new_start_time, new_end_time, score, class_idx]

    # 1. gt
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for vid_name in gt_dict:
        R = [obj for obj in gt_dict[vid_name] if obj[-1] == classidx]
        bbox = np.array([x[:2] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[vid_name] = {'bbox': bbox,
                                'det': det}
    
    # 2. read and parse pred
    # pred lines: [video_name, new_start_time, new_end_time, score, class_idx]
    pred = [x for x in test_preds if x[-1] == classidx and x[3] > score_thres]
    vid_names = [x[0] for x in pred]
    score = np.array([x[3] for x in pred])
    conf_union = score
    PB = np.array([[x[1], x[2]] for x in pred])
    # print(PB)

    # sort the preds by confidence
    sorted_ind = np.argsort(-conf_union)
    try:
    	PB = PB[sorted_ind, :]
    except:
    	return 0., 0., 0.
    vid_names = [vid_names[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(vid_names)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[vid_names[d]]
        pb = PB[d, :]
        ovmax = -np.inf
        BBGT = R['bbox']

        if BBGT.size > 0:
            # calc iou
            mins = np.maximum(pb[0], BBGT[:, 0])
            maxs = np.minimum(pb[1], BBGT[:, 1])
            inters = np.maximum(0., maxs-mins)
            
            area1 = pb[1] - pb[0]
            area2 = BBGT[:, 1] - BBGT[:, 0]

            iou = inters / (area1 + area2 - inters)

            ovmax = np.max(iou)
            jmax = np.argmax(iou)
        
        if ovmax > iou_thres:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.

        else:
            fp[d] = 1.  

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    recall = tp[-1] / float(npos + 1e-10)
    precision = tp[-1] / float(nd + 1e-10)
    return recall, precision, ap

class Config(object):
    def __init__(self):
        self._ordered_keys = []
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key not in ['_ordered_keys'] + self._ordered_keys:
            self._ordered_keys.append(key)
    
    def items(self):
        return [(k, self.__dict__[k]) for k in self._ordered_keys]

    def save_config(self):
        self.config_path = self.save_base + 'config.txt'

        print('=====' * 20)
        print('Config infomation:\n')

        with open(self.config_path, 'w') as f:
            for item in self.items():
                line = '{} = {}'.format(item[0], item[1])
                print(line)
                f.write(line+'\n')
            cur_time = str(datetime.now())
            f.write('\nSave time: {}'.format(cur_time))
        print('\nConfig file has been saved to {}'.format(self.config_path))
        print('=====' * 20)
