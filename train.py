# coding: utf-8

import os, json, pdb
import random
from utils.dataset_utils import get_dataloader
from model.OsalModel import OsalModel

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_from_checkpoint", action='store_true', default=False)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

# GPU setting.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"            

# Basic test.
print("Pytorch's version is {}.".format(torch.__version__))
print("CUDNN's version is {}.".format(torch.backends.cudnn.version()))
print("CUDA's state is {}.".format(torch.cuda.is_available()))
print("CUDA's version is {}.".format(torch.version.cuda))
print("GPU's type is {}.".format(torch.cuda.get_device_name(0)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Currently using device: {}".format(device))

if __name__ == "__main__":

    config_path = './config/config.json'
    try:
        f = open(config_path)
        config = json.load(f)
    except IOError:
        print('Model Building Error: errors occur when loading config file from '+config_path)

    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])

    model = OsalModel()
    model = nn.DataParallel(model).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['step_gamma'])

    if args.train_from_checkpoint: 
        checkpoint = torch.load(opt.checkpoint_path + '9_param.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 1

    train_dataloader = get_dataloader(config, 'training', batch_size=2, shuffle=True, num_worker=4)
    train_dataloader = get_dataloader(config, 'validation', batch_size=1, shuffle=False, num_worker=4)


    # train_dataset = MyDataset(opt)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, 
    #                                                num_workers=opt.num_workers, pin_memory=True)

    # valid_dataset = MyDataset(opt)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, 
    #                                                num_workers=opt.num_workers, pin_memory=True)    

    valid_best_loss = float('inf')

    for epoch in tqdm(range(start_epoch, args.epochs + 1)):

        # Train.
        model.train()
        torch.cuda.empty_cache()
        epoch_train_loss = 0

        for train_iter, (raw_feature, cls_gt, duration_list) in tqdm(enumerate(train_dataloader, start=1)):
            
            optimizer.zero_grad()
            raw_feature = raw_feature.to(device)
            cls_list, reg_list, cls_list_final, reg_list_final = model(raw_feature)
            
            # video_feature, gt_iou_map, start_score, end_score = train_data
            # video_feature = video_feature.cuda()
            # gt_iou_map = gt_iou_map.cuda()
            # start_score = start_score.cuda()
            # end_score = end_score.cuda()

            pdb.set_trace()

            # TODO:finish mask and loss
            bm_mask = get_mask(opt.temporal_scale).cuda()
            # train_loss: total_loss, tem_loss, pem_reg_loss, pem_cls_loss
            train_loss = bmn_loss(bm_confidence_map, start, end, gt_iou_map, start_score, end_score, bm_mask)
            
            train_loss[0].backward()
            optimizer.step()

            epoch_train_loss = epoch_train_loss + train_loss[0].item()

        scheduler.step()

        # Valid.
        epoch_valid_loss = 0
        with torch.no_grad():
            model.eval()
            for valid_iter, valid_data in enumerate(valid_dataloader, start=1):
                video_feature, gt_iou_map, start_score, end_score = valid_data
                video_feature = video_feature.cuda()
                gt_iou_map = gt_iou_map.cuda()
                start_score = start_score.cuda()
                end_score = end_score.cuda()

                bm_confidence_map, start, end = model(video_feature)

                valid_loss = bmn_loss(bm_confidence_map, start, end, gt_iou_map, start_score, end_score, bm_mask)

                epoch_valid_loss = epoch_valid_loss + valid_loss[0].item()

        if epoch <= 10 or epoch % 5 == 0:
            print('Epoch {}: Training loss {:.3}, Validation loss {:.3}'.format(
                    epoch, float(epoch_train_loss/train_iter), float(epoch_valid_loss/valid_iter)))  
            with open(opt.save_path + start_time + '/log.txt', 'a') as f:
                f.write('Epoch {}: Training loss {:.3}, Validation loss {:.3} \n'.format(
                    epoch, float(epoch_train_loss/train_iter), float(epoch_valid_loss/valid_iter)))  


        if epoch_valid_loss < valid_best_loss:
            # Save parameters.
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, opt.save_path + start_time + '/' + str(epoch) + '_param.pth.tar')
            valid_best_loss = epoch_valid_loss
            
            # Save whole model.
            # torch.save(model, opt.save_path)
