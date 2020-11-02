# coding: utf-8
import os
import json
import pdb
import random
from utils.dataset_utils import get_dataloader
from model.OsalModel import OsalModel
from utils.loss_utils import LossCalculator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import os.path as osp
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--train_from_checkpoint", action='store_true', default=False)
parser.add_argument("--train_upsample",action='store_true', default=False)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--name", type=str, default='loss_down_2')
args = parser.parse_args()

# GPU setting.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            

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
        print('Model Building Error: errors occur when loading config file from ' + config_path)
        raise IOError

    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])
    if not osp.exists(osp.join('./reports', args.name)):
        os.makedirs(osp.join('./reports', args.name))
    if not osp.exists(osp.join(config['checkpoint_dir'], args.name)):
        os.makedirs(osp.join(config['checkpoint_dir'], args.name))

    report_dir = osp.join('./reports', args.name)
    checkpoint_dir = osp.join(config['checkpoint_dir'], args.name)
    
    model = OsalModel().to(device)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['step_gamma'])
    # model = nn.DataParallel(model).cuda()
    lossfn = LossCalculator(config, device)

    # if not args.train_upsample:
    #     optimizer = optim.Adam(model.get_ds_param(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # else:
    #     weight_dir = config['checkpoint_dir']
    #     weight_path = osp.join(weight_dir, 'down_sample_3/epoch9_15.118948492705318_adam_param.pth.tar')
    #     checkpoint = torch.load(weight_path)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer = optim.Adam(model.get_us_param(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    #     model.up_sample_init()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['step_gamma'])

    # load_model
    start_epoch = 0
    if args.train_from_checkpoint:
        weight_dir = config['checkpoint_dir']
        weight_path = osp.join(weight_dir, 'epoch10_5.03821693559992_param.pth.tar')
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('checkpoint loaded')

    train_dataloader = get_dataloader(config, 'training', batch_size=args.bs, shuffle=True, num_worker=8)
    valid_dataloader = get_dataloader(config, 'validation', batch_size=args.bs, shuffle=False, num_worker=8)

    valid_best_loss = float('inf')
    clip_len_list = config['feature_lens']
    train_loss_list = []
    # loss_labels = ['background', 'down_sample_regression', 'down_sample_classification', 'down_sample_centerness']
    loss_labels = ['down_sample_regression', 'down_sample_classification', 'down_sample_centerness']
    losses_list = [[], [], [], []]
    valid_idx = []
    valid_loss = []
    valid_losses_list = [[], [], [], []]
    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        # Train.
        model.train()
        torch.cuda.empty_cache()
        epoch_train_loss = 0
        epoch_cls_loss = 0 
        epoch_reg_loss = 0
        reg_loss_fn = torch.nn.MSELoss()
        cls_loss_fn = torch.nn.BCEWithLogitsLoss()
        neg_sample_ratio = 0.2
        pbar = tqdm(train_dataloader)
        cnt = 0
        for raw_feature, cls_gt, duration_list, video_names in pbar:
            # cnt+=1
            # if cnt == 30:
            #     break
            
            optimizer.zero_grad()
            raw_feature = raw_feature.to(device)
            cls_list_final, reg_list_final = model(raw_feature)

            loss, debug_loss = lossfn.calc_loss(cls_list_final, reg_list_final, cls_gt, duration_list, args.train_upsample, epoch)

            loss.backward()
            optimizer.step() 

            if len(train_loss_list)==0:
                train_loss_list.append(loss.detach().cpu().item())
                for index, label in enumerate(loss_labels):
                    losses_list[index].append(debug_loss[index].detach().cpu().item())
            else:
                new_loss = train_loss_list[-1]*0.9 + loss.detach().cpu().item()*0.1
                train_loss_list.append(new_loss)
                for index, label in enumerate(loss_labels):
                    new_loss_ = losses_list[index][-1]*0.9 + 0.1*debug_loss[index].detach().cpu().item()
                    losses_list[index].append(new_loss_)
            pbar.set_description('loss={:.4f} reg={:4f}'.format(loss.detach().cpu().item(), debug_loss[1].detach().cpu().item()))

            if len(train_loss_list)%10 == 0 and len(train_loss_list)!=0:
                # plt.close()
                plt.figure()
                plt.plot(train_loss_list)
                plt.savefig(osp.join(report_dir,'loss_res_adam.jpg'))
                plt.close()

                for index, label in enumerate(loss_labels):
                    plt.figure()
                    plt.plot(losses_list[index])
                    plt.savefig(osp.join(report_dir,'loss_res_{}_adam.jpg'.format(label)) )
                    plt.close()
            
        scheduler.step()

        # Valid.
        epoch_valid_loss = 0
        loss_list = []
        valid_losses_list_ = [[], [], [], []]
        total = 300
        with torch.no_grad():
            model.eval()
            pbar = tqdm(valid_dataloader)
            cnt = 0
            for raw_feature, cls_gt, duration_list, video_names in pbar:
                # cnt+=1
                # # if cnt==1181:
                # if cnt==11:
                #     break

                raw_feature = raw_feature.to(device)
                cls_list_final, reg_list_final = model(raw_feature)

                loss, debug_loss = lossfn.calc_loss(cls_list_final, reg_list_final, cls_gt, duration_list, args.train_upsample, epoch)
                # pdb.set_trace()
                loss_list.append(loss.detach().item())

                for index, label in enumerate(loss_labels):
                    valid_losses_list_[index].append(debug_loss[index].detach().cpu().item())

                pbar.set_description('loss={}'.format(loss.detach().item()))
            
        print('Epoch {}: , Validation loss {:.3}'.format(epoch, np.array(loss_list).mean()))  
        epoch_valid_loss_ = np.array(loss_list).mean()
        valid_idx.append(len(train_loss_list))
        valid_loss.append(np.array(loss_list).mean())
        plt.figure()
        plt.plot(train_loss_list)
        plt.plot(valid_idx, valid_loss)
        plt.savefig(osp.join(report_dir,'loss_res_with_valid_adam.jpg'))
        plt.close()

        for index, label, in enumerate(loss_labels):
            epoch_valid_loss = np.array(valid_losses_list_[index]).mean()
            valid_losses_list[index].append(epoch_valid_loss)
            plt.figure()
            plt.plot(losses_list[index])
            plt.plot(valid_idx, valid_losses_list[index])
            plt.savefig(osp.join(report_dir,'loss_res_valid_{}_adam.jpg'.format(label)))
            plt.close()


        # if epoch_valid_loss_ < valid_best_loss:
        if True:
            # Save parameters.
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, checkpoint_dir+'/epoch' + str(epoch) + '_{}'.format(np.array(loss_list).mean()) + '_adam_param.pth.tar')
            valid_best_loss = epoch_valid_loss_
            
            # Save whole model.
            # torch.save(model, opt.save_path)
