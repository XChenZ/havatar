import torch
import torch.nn as nn
import math

class UnetNoCond5DS(nn.Module):
    # 5DS: downsample 5 times, for posmap size=32
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False,
                return_lowres=False, return_2branches=False):
        super(UnetNoCond5DS, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.return_lowres = return_lowres
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False)

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512
        self.upconv3 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512

        # Coord regressor
        self.upconv4 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 16
        self.upconv5 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode=up_mode) # 32

        if return_2branches:
            self.upconvN4 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 16
            self.upconvN5 = UpConv2DBlock(1 * nf * 2, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode='upconv') # 32

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        # print(x.shape, d1.shape, d2.shape, d3.shape, d4.shape, d5.shape)

        u1 = self.upconv1(d5, d4)
        u2 = self.upconv2(u1, d3)
        u3 = self.upconv3(u2, d2)

        u4 = self.upconv4(u3, d1)
        u5 = self.upconv5(u4)

        if self.return_2branches:
            uN4 = self.upconvN4(u3, d1)
            uN5 = self.upconvN5(uN4)
            return u5, uN5

        return u5


class UnetNoCond7DS_upscale(nn.Module): # 相比于UnetNoCond7DS_new，修正了decoder中复用的第三层，并且最终分辨率高于输入分辨率
    def __init__(self, input_nc=3, output_nc=3, nf=64, up_mode='upconv', use_dropout=False, upscale=False,
                 return_2branches=False):
        super(UnetNoCond7DS_upscale, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.upscale = upscale
        self.return_2branches = return_2branches

        self.conv1 = Conv2DBlock(input_nc, nf, 4, 2, 1, use_bias=False, use_bn=False, use_relu=False)   # //2
        self.conv2 = Conv2DBlock(1 * nf, 2 * nf, 4, 2, 1, use_bias=False, use_bn=True)  # //4
        self.conv3 = Conv2DBlock(2 * nf, 4 * nf, 4, 2, 1, use_bias=False, use_bn=True)  # //8
        self.conv4 = Conv2DBlock(4 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)  # //16
        self.conv5 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)  # //32
        self.conv6 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=True)  # //64
        self.conv7 = Conv2DBlock(8 * nf, 8 * nf, 4, 2, 1, use_bias=False, use_bn=False) # //128

        self.upconv1 = UpConv2DBlock(8 * nf, 8 * nf, 4, 2, 1, up_mode=up_mode) #2x2, 512    # //64
        self.upconv2 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 4x4, 512  # //32
        self.upconv3 = UpConv2DBlock(8 * nf * 2, 8 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout) # 8x8, 512  # //16

        # Coord regressor
        self.upconvC4 = UpConv2DBlock(8 * nf * 2, 4 * nf, 4, 2, 1, up_mode=up_mode, use_dropout=use_dropout)  # 8x8, 512
        self.upconvC5 = UpConv2DBlock(4 * nf * 2, 2 * nf, 4, 2, 1, up_mode=up_mode) # 16 # //8
        self.upconvC6 = UpConv2DBlock(2 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 32 # //4
        self.upconvC7 = UpConv2DBlock(1 * nf * 2, 1 * nf, 4, 2, 1, up_mode=up_mode) # 64x64, 128
        if upscale:
            self.upconvC8 = UpConv2DBlock(1 * nf, output_nc, 4, 2, 1, use_bn=False, use_bias=True, up_mode=up_mode)  # 64x64, 128

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)
        # print(x.shape, d1.shape, d2.shape, d3.shape, d4.shape, d5.shape, d6.shape, d7.shape)

        # shared decoder layers
        u1 = self.upconv1(d7, d6)
        u2 = self.upconv2(u1, d5)
        u3 = self.upconv3(u2, d4)

        # coord regressor
        uc4 = self.upconvC4(u3, d3)
        uc5 = self.upconvC5(uc4, d2)
        uc6 = self.upconvC6(uc5, d1)
        uc7 = self.upconvC7(uc6)
        out = self.upconvC8(uc7) if self.upscale else uc7

        return out


class simple_Decoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, input_res=16, output_res=256, up_mode='upconv'):
        super(simple_Decoder, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16,
        }

        self.layers = nn.ModuleList()

        if True:
            res = int(input_res * 2)
            self.layers.append(UpConv2DBlock(input_nc, self.channels[res], 4, 2, 1, up_mode=up_mode, use_dropout=False))
        else:
            res = int(input_res)
            self.layers.append(Conv2DBlock(input_nc, self.channels[res], 4, 1, 1, use_bias=True, use_bn=True, use_relu=False))

        for l in range(int(math.log2(output_res) - math.log2(input_res)) - 2):
            self.layers.append(UpConv2DBlock(self.channels[res], self.channels[res*2], 4, 2, 1, up_mode=up_mode, use_dropout=False))
            res = res * 2
        # if output_nc ==
        assert output_nc == self.channels[output_res]
        self.layers.append(UpConv2DBlock(self.channels[res], self.channels[res*2], 4, 2, 1, up_mode=up_mode, use_dropout=False, use_bn=False))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, use_bias=False, use_bn=True, use_relu=True):
        super(Conv2DBlock, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        if use_bn:
            # self.bn = nn.BatchNorm2d(output_nc, affine=False)
            self.bn = nn.InstanceNorm2d(output_nc, affine=False)    ### by zxc
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.use_relu:
            x = self.relu(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class UpConv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1,
                 use_bias=False, use_bn=True, up_mode='upconv', use_dropout=False):
        super(UpConv2DBlock, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.relu = nn.ReLU()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=use_bias)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, stride=1),
            )
        if use_bn:
            # self.bn = nn.BatchNorm2d(output_nc, affine=False)
            self.bn = nn.InstanceNorm2d(output_nc, affine=False)
        if use_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x, skip_input=None):
        x = self.relu(x)
        x = self.up(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.drop(x)
        if skip_input is not None:
            x = torch.cat([x, skip_input], 1)
        return x
