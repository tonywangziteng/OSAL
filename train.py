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
from collections import OrderedDict
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("--train_from_checkpoint", action='store_true', default=False)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, default=16)
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
    
    model = OsalModel().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['step_gamma'])
    # model = nn.DataParallel(model).cuda()
    lossfn = LossCalculator(config, device)

    # load_model
    start_epoch = 0
    if args.train_from_checkpoint:
        weight_dir = config['checkpoint_dir']
        weight_path = osp.join(weight_dir, 'epoch3_6.566827012802744_param.pth.tar')
        checkpoint = torch.load(weight_path)

        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v

        # model.load_state_dict(new_state_dict)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['step_gamma'])

    train_dataloader = get_dataloader(config, 'training', batch_size=args.bs, shuffle=True, num_worker=8)
    valid_dataloader = get_dataloader(config, 'validation', batch_size=8, shuffle=False, num_worker=8)


    # train_dataset = MyDataset(opt)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, 
    #                                                num_workers=opt.num_workers, pin_memory=True)

    # valid_dataset = MyDataset(opt)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, 
    #                                                num_workers=opt.num_workers, pin_memory=True)    

    valid_best_loss = float('inf')
    clip_len_list = config['feature_lens']
    train_loss_list = []
    debug_loss_list = []
    valid_idx = []
    valid_loss = []
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
            cnt+=1
            if cnt == 603:
                break
            
            optimizer.zero_grad()
            raw_feature = raw_feature.to(device)
            cls_list, reg_list, cls_list_final, reg_list_final = model(raw_feature)

            loss, debug_loss = lossfn.calc_loss(cls_list, reg_list, cls_list_final, reg_list_final, cls_gt, duration_list, epoch)

            # pdb.set_trace()
            # try:
            loss.backward()
            optimizer.step() 
            # debug_loss_list.append(debug_loss.detach().cpu().item())
            # except:
            #     print('something wrong happened, debug_loss is: ', debug_loss)

            if len(train_loss_list)==0:
                train_loss_list.append(loss.detach().cpu().item())
            else:
                new_loss = train_loss_list[-1]*0.9 + loss.detach().cpu().item()*0.1
                train_loss_list.append(new_loss)
            pbar.set_description('loss={:.4f}'.format(loss.detach().cpu().item()))

            if len(train_loss_list)%3 == 0 and len(train_loss_list)!=0:
                plt.close()
                plt.figure()
                plt.plot(train_loss_list)
                plt.savefig('loss_res3.jpg')

            # print(model.state_dict()['up_sample_head.cls_head.bias'])
            # print(video_names)
            for a in model.state_dict().values():
                if torch.isnan(a).any():
                    pdb.set_trace()

                # plt.close()
                # plt.figure()
                # plt.plot(debug_loss_list)
                # plt.savefig('debug_res.jpg')
            
        scheduler.step()

        # Valid.
        epoch_valid_loss = 0
        loss_list = []
        total = 300
        with torch.no_grad():
            model.eval()
            pbar = tqdm(valid_dataloader)
            cnt = 0
            for raw_feature, cls_gt, duration_list, video_names in pbar:
                # cnt+=1
                # if cnt==total:
                #     break

                raw_feature = raw_feature.to(device)
                cls_list, reg_list, cls_list_final, reg_list_final = model(raw_feature)

                loss, _ = lossfn.calc_loss(cls_list, reg_list, cls_list_final, reg_list_final, cls_gt, duration_list, epoch)

                loss_list.append(loss.detach().item())
                
                pbar.set_description('loss={}'.format(loss.detach().item()))
            
        print('Epoch {}: , Validation loss {:.3}'.format(epoch, np.array(loss_list).mean()))  
        epoch_valid_loss = np.array(loss_list).mean()
        valid_idx.append(len(train_loss_list))
        valid_loss.append(np.array(loss_list).mean())
        plt.figure()
        plt.plot(train_loss_list)
        plt.plot(valid_idx, valid_loss)
        plt.savefig('loss_res_with_valid3.jpg')
        plt.close()


        if epoch_valid_loss < valid_best_loss:
            # Save parameters.
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, config['checkpoint_dir']+'/epoch' + str(epoch) + '_{}'.format(np.array(loss_list).mean()) + '_param.pth.tar')
            valid_best_loss = epoch_valid_loss
            
            # Save whole model.
            # torch.save(model, opt.save_path)
