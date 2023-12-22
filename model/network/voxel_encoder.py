'''A CycleGAN Encoder'''

import torch.nn as nn
import torch
import math

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initializes network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Residual3D(BaseNetwork):
    def __init__(self, numIn, numOut):
        super(Residual3D, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.GroupNorm(4, self.numIn)  ###?????????
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(self.numIn, self.numOut, bias=True, kernel_size=3, stride=1,
                               padding=2, dilation=2)
        self.bn1 = nn.GroupNorm(4, self.numOut)
        self.conv2 = nn.Conv3d(self.numOut, self.numOut, bias=True, kernel_size=3, stride=1,
                               padding=1)
        self.bn2 = nn.GroupNorm(4, self.numOut)
        # self.conv3 = nn.Conv3d(self.numOut, self.numOut, bias=True, kernel_size=3, stride=1,
        #                        padding=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv3d(self.numIn, self.numOut, bias=True, kernel_size=1)
        self.init_weights()

    def forward(self, x):
        residual = x
        # out = self.bn(x)
        # out = self.relu(out)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.conv3(out)
        # out = self.relu(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        out = self.relu(out + residual)
        return out


class VolumeEncoder(BaseNetwork):
    """CycleGan Encoder"""
    def __init__(self, num_in=3, num_inter=32, num_out=128):
        super(VolumeEncoder, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.num_inter = num_inter
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv3d(self.num_in, self.num_inter, bias=True, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn0 = nn.GroupNorm(4, self.num_inter)

        self.r1 = Residual3D(self.num_inter, self.num_inter)
        self.conv1 = nn.Conv3d(self.num_inter, self.num_inter, bias=True, kernel_size=3, stride=2, padding=4, dilation=2)
        self.bn1 = nn.GroupNorm(4, self.num_inter)

        self.r2 = Residual3D(self.num_inter, self.num_inter)
        self.conv2 = nn.Conv3d(self.num_inter, self.num_inter, bias=True, kernel_size=3, stride=2, padding=4, dilation=2)
        self.bn2 = nn.GroupNorm(4, self.num_inter)

        self.r3 = Residual3D(self.num_inter, self.num_inter)
        self.conv3 = nn.Conv3d(self.num_inter, self.num_inter, bias=True, kernel_size=3, stride=2, padding=4, dilation=2)
        self.bn3 = nn.GroupNorm(4, self.num_inter)

        # self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))
        # self.fc = nn.Linear(8 * self.num_inter, self.num_out)

        self.init_weights()

    def forward(self, x, intermediate_output=True):
        out = x
        out = self.conv0(out)
        out = self.bn0(out)
        inter_volume_feat_ls = []
        ## 32,32 -> 16,32
        out = self.r1(out)
        if intermediate_output:
            inter_volume_feat_ls.append(out.clone())
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        ## 16,32 -> 8,32
        out = self.r2(out)
        if intermediate_output:
            inter_volume_feat_ls.append(out.clone())
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        ## 8,32 -> 4,32
        out = self.r3(out)
        if intermediate_output:
            inter_volume_feat_ls.append(out.clone())
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)


        ## 4,32 -> 2,32 -> 256 -> 128
        out = self.avgpool(out)  # [B, C, D, H, W]
        out = torch.flatten(out, 1)  # [B, L]
        out = self.fc(out)

        if intermediate_output:
            return out, inter_volume_feat_ls
        else:
            return out  # [B, L]


class VolumeDecoder(BaseNetwork):
    """CycleGan Encoder"""
    def __init__(self, num_in=1024, num_out=1, final_res=32, up_mode='upsample'):
        super(VolumeDecoder, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.relu = nn.ReLU(inplace=True)

        self.register_buffer('init_lc', torch.rand(1, num_in, 1, 1, 1)) # [B, C, H, W, D]
        self.filters = nn.ModuleList()
        num_layers = int(math.log2(final_res))
        init_log2 = int(math.log2(num_in))
        for i in range(num_layers):# [1024, 512, 256, 128, 64, 32]
            self.filters.append(UpConv3DBlock(input_nc=2**(init_log2-i), output_nc=2**(init_log2-i-1), up_mode=up_mode))
        self.final_conv = nn.Conv3d(2**(init_log2-num_layers), num_out, bias=True, kernel_size=3, padding=1, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))
        # self.fc = nn.Linear(8 * self.num_inter, self.num_out)
        self.init_weights()

    def forward(self):
        x = self.init_lc
        for f in self.filters:  # [1, 1024] -> [32, 32]
            x = self.relu(f(x))

        # out = torch.nn.functional.softmax(self.final_conv(x)) ## 32,32 -> 32,2
        x = torch.sigmoid(self.final_conv(x))

        # x[:, :, :, 0, :] = 0.; x[:, :, :1, :x.shape[-1] // 8, :] = 0.   # # 因为目前我的skinning field只开在脖子附近，因此需要保证skinning filed的最上层（及以上的头的部分）都跟随头部姿态转动
        out = torch.cat([x, 1-x], dim=1)
        return out



class UpConv3DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1,
                 use_bias=True, use_bn=True, up_mode='upconv', use_dropout=False):
        super(UpConv3DBlock, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(input_nc, output_nc, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=use_bias)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='trilinear', scale_factor=2, align_corners=False),
                nn.Conv3d(input_nc, output_nc, kernel_size=3, padding=1, stride=1),
            )
        self.norm = nn.InstanceNorm3d(output_nc, affine=False)

        if use_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x, skip_input=None):
        x = self.up(x)
        x = self.norm(x)

        if self.use_dropout:
            x = self.drop(x)
        if skip_input is not None:
            x = torch.cat([x, skip_input], 1)
        return x