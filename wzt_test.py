from utils.dataset_utils import get_dataloader
from model.OsalModel import OsalModel
import json

import pdb

if __name__ == '__main__':
    config_path = './config/config.json'
    try:
        f = open(config_path)
        config = json.load(f)
    except IOError:
        print('Model Building Error: errors occur when loading config file from '+config_path)
    data_loader = get_dataloader(config, 'training', batch_size=2)
    OSAL = OsalModel()
    for idx, (raw_feature, cls_gt, duration_list) in enumerate(data_loader):
        cls_list, reg_list, cls_list_final, reg_list_final = OSAL(raw_feature)
        pdb.set_trace()