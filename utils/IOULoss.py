import torch
import torch.nn as nn
import pdb

class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()
    def iouloss_calc(self, reg_pred, reg_gt, batch_size):
        for layer in range(5):
            reg_pred_inbatch = reg_pred[layer].cpu()
            reg_gt_inbatch = reg_gt[layer]
            iouloss_list = []
            for batch in range(batch_size):
                reg_result = reg_pred_inbatch[batch]
                reg_target = reg_gt_inbatch[batch]
                for reg_target_inclip in reg_target:
                    start_time = reg_result[: , 0]
                    end_time = reg_result[: , 1]
                    iou0 = torch.where((start_time < reg_target_inclip[0]) & (end_time > reg_target_inclip[1]), (reg_target_inclip[1] - reg_target_inclip[0])/ (end_time - start_time), torch.tensor(0.))
                    iou1 = torch.where((start_time < reg_target_inclip[0]) & (end_time < reg_target_inclip[1]) & (end_time > reg_target_inclip[0]), (end_time - reg_target_inclip[0])/ (reg_target_inclip[1] - start_time), torch.tensor(0.))
                    iou2 = torch.where((start_time > reg_target_inclip[0]) & (start_time < reg_target_inclip[1]) & (end_time > reg_target_inclip[1]), (reg_target_inclip[1] - start_time)/ (end_time -reg_target_inclip[0]), torch.tensor(0.))
                    iou = iou0 + iou1 + iou2
                    iou = torch.where(iou < 1e-6, torch.tensor(1.), iou)
                    loss = torch.sum(- torch.log(iou))
                    iouloss_list.append(loss)
                    pdb.set_trace()
        return sum(iouloss_list) / batch_size

if __name__ == "__main__":
    IOUFn = IOULoss()
    a = (torch.tensor(([[
        [1,6],[6,9],[1,9],[1,2]
    ]]), dtype=torch.float32).abs())
    b = (torch.tensor(([[
        [3,8],[0,2]
    ]]), dtype=torch.float32).abs())
    reg_result = [a,a,a,a,a]
    reg_target = [b,b,b,b,b]
    print(IOUFn.iouloss_calc(reg_result, reg_target, 1))
