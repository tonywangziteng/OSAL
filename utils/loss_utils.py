import torch
import numpy as np 

class LossCalculator():
    def __init__(self, config):
        cfg = config('loss')

    def calc_loss(
        self, cls_list, reg_list, cls_list_final, reg_list_final, 
        cls_gt, duration_list
    ):
        # calculate classification loss
        
        pass

    def get_reg_gt(self, duration_list):
        pass
