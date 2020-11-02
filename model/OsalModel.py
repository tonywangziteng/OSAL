import torch
import torch.nn as nn
import json, pdb, math
import itertools
from collections import OrderedDict
from tensorboardX import SummaryWriter

class Scaler(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scaler, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
    def forward(self, input):
        return input * self.scale       

class DownSample(nn.Module):
    """
    Down sample process model
    param:  
    :config: (dict) whole config dictionary loaded from config file
    """
    def __init__(self, config):
        super(DownSample, self).__init__()
        cfg = config['down_sample']
        print(cfg)
        kernel_sizes = cfg['kernel_size']
        out_dims = cfg['dims']
        in_dim = config['feature_dim']
        for idx, out_dim in enumerate(out_dims):
            self.add_module(
                'down_sample_{}'.format(idx),
                nn.Sequential(
                    
                    nn.Conv1d(in_dim, out_dim, kernel_size=kernel_sizes[idx], padding=1),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(inplace=True),
                    
                    nn.Conv1d(out_dim, out_dim, kernel_size=kernel_sizes[idx], padding=1),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(inplace=True),
                    
                    nn.Conv1d(out_dim, out_dim, kernel_size=kernel_sizes[idx], padding=1, stride=2),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(inplace=True)
                )
            )
            in_dim = out_dim
        self.submodule_names = list(self._modules.keys())

        # he initialization
        for module in self.modules():
            # pdb.set_trace()
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in')

    def forward(self, feature):
        feature_list = []
        x = feature
        # feature_list.append(x)
        for name in self.submodule_names:
            submodule = getattr(self, name)
            x = submodule(x)
            feature_list.append(x)
        # pdb.set_trace()
        return feature_list

class DownSampleHead(nn.Module):
    """
    Process after coarse down sample
    make every layer of input to the same chammel number 
    and then get the classification result and regression result using a shared network
    """
    def __init__(self, config):
        super(DownSampleHead, self).__init__()
        cfg = config['down_sample_head']
        layer_dims = cfg['in_dims']
        out_dim = cfg['out_dim']
        self.name_list = []
        for idx in range(len(layer_dims)):
            in_dim = layer_dims[idx]
            self.name_list.append('truncate_{}'.format(idx))
            self.add_module(
                'truncate_{}'.format(idx), 
                nn.Sequential(
                    nn.BatchNorm1d(in_dim), 
                    # nn.InstanceNorm1d(in_dim),
                    nn.Conv1d(in_dim, out_dim, kernel_size=1),
                    nn.ReLU(inplace=True)
                )
            )

        # shared head for cls and reg
        # cls output is 201 dim, cls number + bg
        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(out_dim), 
            nn.Conv1d(out_dim, out_dim, kernel_size=1, padding=0), 
            nn.Tanh(), 
            nn.Conv1d(out_dim, 200, kernel_size=1, padding=0)
        )
        
        self.bg_head = nn.Sequential(
            nn.BatchNorm1d(out_dim), 
            nn.Conv1d(out_dim, out_dim//2, kernel_size=1, padding=0), 
            nn.Tanh(), 
            nn.Conv1d(out_dim//2, 1, kernel_size=1, padding=0)
        )
        
        # start, end , centerness
        self.reg_head = nn.Sequential(
            nn.BatchNorm1d(out_dim), 
            nn.Conv1d(out_dim, out_dim, kernel_size=1, padding=0), 
            nn.Tanh(), 
            nn.Conv1d(out_dim, 3, kernel_size=1, padding=0)
        )

        self.scales = nn.ModuleList([Scaler(init_value=1.0) for _ in range(5)])

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')        
        nn.init.constant(self.bg_head.state_dict()['3.bias'], -math.log(99))


    def forward(self, feature_list):
        out_list = []
        for idx, feature in enumerate(feature_list):
            name = self.name_list[idx]
            module = getattr(self, name)
            out = module(feature)
            out_list.append(out)
        
        cls_list, reg_list = [], []
        for l, feature in enumerate(out_list):
            # x = self.scales[l](self.cls_head(feature))
            x = self.cls_head(feature)
            # cls_result = torch.softmax(x[:, :200, :], dim=1)
            cls_result = x
            bg_result = torch.sigmoid(self.bg_head(feature))
            # bg_result = torch.sigmoid(x[:, 200:, :])
            reg_result = self.scales[l](torch.tanh(self.reg_head(feature)))
            cls_list.append(torch.cat([cls_result, bg_result], dim=1))
            # pdb.set_trace()
            reg_list.append(reg_result)

        return cls_list, reg_list


class UpSample(nn.Module):
    def __init__(self, config):
        super(UpSample, self).__init__()
        cfg = config['up_sample']
        in_dim = cfg['in_dim']
        out_dims = cfg['out_dims']
        out_padding = cfg['out_padding']

        for idx, out_dim in enumerate(out_dims):
            self.add_module(
                'up_sample_{}'.format(idx),
                MergeModule(in_dim, out_dim, out_padding[idx])
            )
            in_dim = out_dim
        self.submodule_names = list(self._modules.keys())

        in_dim = cfg['in_dim']
        self.merge_0 = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_dim), 
            nn.LeakyReLU()
        )

        # he initialization
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, input_feature, feature_list):
        out_list = []
        input_feature = self.merge_0(input_feature) + input_feature
        out_list.append(input_feature)
        feature = input_feature
        for idx, module_name in enumerate(self.submodule_names):
            submodule = getattr(self, module_name)
            feature = submodule(feature, feature_list[-idx-2])
            out_list.append(feature)
        return out_list


class MergeModule(nn.Module):
    '''
    Merge the feature from last layer and the feature of Unet structure
    '''
    def __init__(self, in_dim, out_dim, out_padding):
        super(MergeModule, self).__init__()
        self.origin_branch_1 = nn.Sequential(
            nn.ConvTranspose1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )
        self.origin_branch_2 = nn.Sequential(
            nn.ConvTranspose1d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=out_padding), 
            nn.BatchNorm1d(out_dim),
            nn.Tanh()
        )
        # self.origin_branch_2 = nn.ConvTranspose1d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=out_padding)
        self.Unet_branch = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_dim),
            nn.Tanh()
        )
    
        self.merge_conv = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU()
        )
        

    def forward(self, origin_input, Unet_input):
        origin_output = self.origin_branch_1(origin_input)
        origin_output = self.origin_branch_2(origin_output)

        Unet_output = self.Unet_branch(Unet_input)
        merged_input = Unet_output + origin_output
        merged_output = self.merge_conv(merged_input)
        
        return merged_output


class UpSampleHead(nn.Module):
    def __init__(self, config):
        super(UpSampleHead, self).__init__()
        cfg = config['up_sample_head']
        layer_dims = cfg['in_dims']
        out_dim = cfg['out_dim']
        self.name_list = []
        for idx in range(len(layer_dims)):
            in_dim = layer_dims[idx]
            self.name_list.append('truncate_{}'.format(idx))
            self.add_module(
                'truncate_{}'.format(idx), 
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel_size=1),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(inplace=True)
                )
            )
        # shared head for cls and reg
        # cls output is 201 dim, cls number + bg
        # 3 is the number of channels to be concatenated 
        # cls output is 201 dim, cls number + bg
        self.cls_head = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=1, padding=0), 
            nn.BatchNorm1d(out_dim), 
            nn.Tanh(), 
            nn.Conv1d(out_dim, 201, kernel_size=1, padding=0)
        )
        
        # self.bg_head = nn.Sequential(
        #     nn.Conv1d(out_dim, out_dim, kernel_size=1, padding=0), 
        #     nn.BatchNorm1d(out_dim), 
        #     nn.Tanh(), 
        #     nn.Conv1d(out_dim, 1, kernel_size=1, padding=0)
        # )
        
        # start, end , centerness
        self.reg_head = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=1, padding=0), 
            nn.BatchNorm1d(out_dim),
            nn.ReLU(), 
            nn.Conv1d(out_dim, 3, kernel_size=1, padding=0),
        )

        self.scales = nn.ModuleList([Scaler(init_value=1.0) for _ in range(5)])
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')        

    def forward(self, feature_list):
        feature_list = feature_list[::-1]
        out_list = []
        # truncate all the features to a specific channel
        for idx, feature in enumerate(feature_list):
            name = self.name_list[idx]
            module = getattr(self, name)
            out = module(feature)
            out_list.append(out)

        cls_list, reg_list = [], []
        for l, feature in enumerate(out_list):

            # x = self.cls_head(feature)
            x = torch.sigmoid(self.cls_head(feature)) 

            # bg_result = torch.sigmoid(self.bg_head(feature))
            # cls_list.append(torch.cat([x, bg_result], dim=1))
            reg_result = self.scales[l](torch.exp(self.reg_head(feature)))
            cls_list.append(x)
            reg_list.append(reg_result)

        return cls_list, reg_list


class OsalModel(nn.Module):
    def __init__(self):
        super(OsalModel, self).__init__()
        self.temp_scale = 100
        self.input_dim = 400

        config_path = '/home/e813/wzt_code/wzt_OSAL/model/model_cfg.json'
        try:
            f = open(config_path)
            config = json.load(f)
        except IOError:
            print('Model Building Error: errors occur when loading config file from '+config_path)
        self.down_sample = DownSample(config)
        # self.down_sample_head = DownSampleHead(config)
        self.up_sample = UpSample(config)
        self.up_sample_head = UpSampleHead(config)
        print(self)
        # he initialization
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, feature):
        feature_list = self.down_sample(feature)
        # cls_list, reg_list = self.down_sample_head(feature_list)
        out_list = self.up_sample(feature_list[-1], feature_list)
        cls_list_final, reg_list_final = self.up_sample_head(out_list)

        return cls_list_final, reg_list_final

    def get_ds_param(self):
        return itertools.chain(self.down_sample.parameters(), self.down_sample_head.parameters())

    def get_us_param(self):
        return itertools.chain(self.up_sample.parameters(), self.up_sample_head.parameters())

    def up_sample_init(self):
        for m in self.up_sample.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        for m in self.up_sample_head.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

if __name__ == '__main__':
    model = OsalModel()
    input_ = torch.ones((2, 400 ,100))
    writer = SummaryWriter("/home/e813/wzt_code/wzt_OSAL/model/result",comment='OSAL2')
    writer.add_graph(model, input_, True)