from utils.dataset_utils import get_dataloader
from model.OsalModel import OsalModel

import pdb

if __name__ == '__main__':
    data_loader = get_dataloader('training', batch_size=2)
    OSAL = OsalModel()
    for idx, (raw_feature, cls_gt, duration_list) in enumerate(data_loader):
        cls_list, reg_list, cls_list_final, reg_list_final = OSAL(raw_feature)
        pdb.set_trace()