import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import cv2
import torch.nn.init as int
from base_net import *
from CAB import CrossSwinTransformer_1

from MSA import *
from SCExt import PanExt

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor

class block1(nn.Module):
    def __init__(self, n_feats):
        super(block1, self).__init__()
        self.cross_attn = CrossSwinTransformer_1(n_feats, n_heads=4, head_dim=8, win_size=4, cross_module=['ms', 'pan'], cat_feat=['ms', 'pan'])
        # self.conv_SL = ConvBlock(64, 32, 3, 1, 1, activation='relu')
        self.four_conv_1 = four_conv(16, 20)
        self.con_1 = convert_conv(116, 16,3)
        self.five_conv_1 = five_conv(16, 16)
        self.con_2 = convert_conv(132, 16, 3)
        self.four_conv_2 = four_conv(16, 20)
        self.con_3 = convert_conv(232, 16, 3)
        self.five_conv_2 = five_conv(16, 16)
        self.con_4 = convert_conv(248, 16)
        self.conv_out1 = ConvBlock(16, 8, 3, padding=1, stride=1, activation='prelu')
        self.conv_out2 = ConvBlock(8, 8, 3, padding=1, stride=1, activation='prelu')
        init_weights(self.cross_attn, self.four_conv_1, self.con_1, self.five_conv_1, self.con_2, self.four_conv_2, self.con_3, self.five_conv_2,
                     self.con_4, self.conv_out1, self.conv_out2)




    def forward(self, ms, pan):
        f1 = self.cross_attn(ms, pan) #16
        # f1 = self.conv_SL(f1) #16
        f11 = self.four_conv_1(f1) #100
        f2 = torch.cat((f1, f11), dim=1)  #116
        f2 = self.con_1(f2)
        f21 = self.five_conv_1(f2)  #16
        f3 = torch.cat((f1, f11, f21), dim=1)   #132
        f3 = self.con_2(f3)
        f12 = self.four_conv_2(f3)  #100
        f4 = torch.cat((f1, f11, f21, f12), dim=1)  #232
        f4 = self.con_3(f4)
        f22 = self.five_conv_2(f4)
        f5 = torch.cat((f1, f11, f21, f12, f22), dim=1) #248
        f5 = self.con_4(f5)   #16
        hp_pan = pan - F.interpolate(
            F.interpolate(pan, scale_factor=1 / 2, mode='bicubic', align_corners=False, recompute_scale_factor=True),
            scale_factor=2, mode='bicubic', align_corners=False, recompute_scale_factor=True)
        f5 = self.conv_out1(f5)
        x = f5 * hp_pan  #torch.Size([1, 8, 128, 128])
        x = x + ms    #torch.Size([1, 8, 128, 128])
        x = self.conv_out2(x)   #torch.Size([1, 8, 128, 128])
        return x

class PanBranch(nn.Module):
    def __init__(self):
        super(PanBranch, self).__init__()
        #32
        self.conv_input = ConvBlock(1, 32, 3, 1, 1, bias=True, activation='prelu')
        self.RB = ResnetBlock(32)
        self.conv_pan_1 = ConvBlock(32, 8, 3, 1, 1, bias=True, activation='prelu')

        self.ext1 = PanExt(8)
        self.conv_pan_2 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext2 = PanExt(8)

        self.downsample = ConvBlock(8, 8, 2, 2, 0, bias=True, activation='prelu')

        self.ext3 = PanExt(8)
        self.conv_pan_3 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext4 = PanExt(8)

        self.prelu = nn.PReLU(init=0.5)
        self.conv_12 = ConvBlock(32, 32, 3, 1, 1, bias=True, activation='prelu')
        self.conv_22 = ConvBlock(32, 32, 3, 1, 1, bias=True, activation='prelu')
        self.conv_output2 = ConvBlock(32, 8, 3, 1, 1, bias=True, activation='prelu')

        self.conv_hp_1 = ConvBlock(8, 1, 3, 1, 1, bias=True, activation='prelu')
        self.conv_hp_2 = ConvBlock(8, 1, 3, 1, 1, bias=True, activation='prelu')

        # 使用 PixelShuffle 上采样4倍（需先通过卷积扩展通道数）
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(3, 3 * (4 ** 2), kernel_size=3, padding=1),  # 扩展通道数到 3 * 16
            nn.PixelShuffle(4)  # 将通道数转换为空间分辨率
        )
        self.conv_ms = ConvBlock(3, 32, 3, 1, 1, bias=True, activation='prelu')

        # init_weights(self.conv_input, self.conv_1, self.conv_2, self.conv_output1, self.downsample,self.conv_12, self.conv_22, self.conv_output2, self.prelu, self.conv_hp_1, self.conv_hp_2)


    def forward(self, pan, ms):
        ms_up4 = F.interpolate(ms, scale_factor=4, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        ms_up4 = self.conv_ms(ms_up4)

        feature_1 = self.conv_input(pan)  # h*w*32
        feature_1 = feature_1 - ms_up4
        feature_2 = self.RB(feature_1)
        feature_3 = feature_2 + ms_up4
        feature_3 = self.conv_pan_1(feature_3)

        pan_1 = self.ext1(feature_3)  #

        p4 = self.conv_pan_2(pan_1)
        pan_2 = self.ext2(p4)  #

        p5 = self.downsample(feature_3)
        pan_3 = self.ext3(p5)   #
        p6 = self.conv_pan_3(pan_3)
        pan_4 = self.ext4(p6)  #


        hp_pan_1 = pan - F.interpolate(  # high-pass   通过先将图像缩小，再放大来模仿低通滤波的效果。
            F.interpolate(pan, scale_factor=1 / 4, mode='bicubic', align_corners=False, recompute_scale_factor=True),
            scale_factor=4, mode='bicubic', align_corners=False, recompute_scale_factor=True)
        hp_pan_2 = self.conv_hp_1(pan_2) - F.interpolate(
            F.interpolate(self.conv_hp_1(pan_2), scale_factor=1 / 4, mode='bicubic', align_corners=False, recompute_scale_factor=True),
            scale_factor=4, mode='bicubic', align_corners=False, recompute_scale_factor=True)
        hp_pan_3 = self.conv_hp_2(pan_4) - F.interpolate(
            F.interpolate(self.conv_hp_2(pan_4), scale_factor=1 / 2, mode='bicubic', align_corners=False, recompute_scale_factor=True),
            scale_factor=2, mode='bicubic', align_corners=False, recompute_scale_factor=True)

        return pan_1, pan_2, pan_3, pan_4, hp_pan_1, hp_pan_2,hp_pan_3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        upsample = 2
        self.pan_branch = PanBranch()
        self.block1 = block1(8)
        self.block2 = block1(8)
        self.conv_input = ConvBlock(3,32,3, 1, 1, bias=True, activation='prelu')
        self.subpixel_up = nn.PixelShuffle(upsample)
        self.conv_hr1 = ConvBlock(8,3,3, 1, 1, bias=True, activation='prelu')
        self.RB1 = ResnetBlock(8)

        self.ext1 = PanExt(8)
        self.conv_ms_11 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext2 = PanExt(8)
        self.RB2 = ResnetBlock(8)
        self.conv_f_11 = ConvBlock(24, 8, 3, 1, 1, bias=True, activation='prelu')

        self.msa = SCAttention(n_feats=8, n_heads=4, head_dim=8, win_size=4)
        self.conv_f_12 = ConvBlock(8, 32, 3, 1, 1, bias=True, activation='prelu')

        self.conv_ms_21 = ConvBlock(8, 32, 3, 1, 1, bias=True, activation='prelu')
        self.conv_ms_22 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext3 = PanExt(8)
        self.ext4 = PanExt(8)
        self.RB3 = ResnetBlock(8)
        self.RB4 = ResnetBlock(8)


        self.final_conv1 = ConvBlock(24, 8, 3, 1, 1, bias=True, activation='prelu')
        self.final_conv2 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.final_conv3 = ConvBlock(16, 3, 3, 1, 1, bias=True, activation='prelu')
        self.conv_pan = ConvBlock(1, 8, 3, 1, 1, bias=True, activation='prelu')



        # init_weights(self.conv_input, self.conv_21, self.subpixel_up, self.final_conv1, self.final_conv2)



    def forward(self, x_pan, l_ms):

        pan_1, pan_2, pan_3, pan_4, hp_1, hp_2, hp_3= self.pan_branch(x_pan, l_ms)  #1*8*256*256   1*8*128*128   torch.Size([1, 1, 128, 128])
        ms_1 = self.conv_input(l_ms)    #torch.Size([1, 32, 64, 64])
        ms_b1 = self.subpixel_up(ms_1)    #torch.Size([1, 8, 128, 128])
        f1 = self.block1(ms_b1, pan_4)  # torch.Size([1, 8, 128, 128])
        f1 = self.RB1(f1)

        ms_11 = self.ext1(ms_b1)   #

        f2 = f1 + ms_11 + hp_3 + pan_4
        f2 = self.RB2(f2)



        ms_12 = self.conv_ms_11(ms_11)
        ms_13 = self.ext2(ms_12)   #
        f3 = torch.cat([f2, ms_13, pan_3], dim=1)
        f3 = self.conv_f_11(f3)
        f3 = self.msa(f3)
        f3 = self.conv_f_12(f3)

        f3 = self.subpixel_up(f3)
        hrms1 = self.conv_hr1(f3)
        f4 = self.block1(f3, pan_2)    #ms和pan进行cab，然后加f3

        ms_2 = self.conv_ms_21(ms_b1)
        ms_2 = self.subpixel_up(ms_2)
        # print(ms_2.shape)
        ms_21 = self.ext3(ms_2)  #
        f5 = f4 + ms_21 + hp_2 + pan_2
        f5 = self.RB3(f5)
        ms_22 = self.conv_ms_22(ms_21)
        ms_23 = self.ext4(ms_22)  #

        # print(f5.shape, ms_23.shape, pan_1.shape)
        f6 = torch.cat([f5, ms_23, pan_1], dim=1)

        f7 = self.final_conv1(f6)    #torch.Size([1, 3, 256, 256])

        f7 = self.msa(f7) +hp_1
        f7 = self.final_conv2(f7)

        x_pan = self.conv_pan(x_pan)
        fusion_1 = self.block2(f3, x_pan)
        fusion_2 = self.block2(f7, x_pan)
        hrms = torch.cat([fusion_1, fusion_2], dim=1)
        hrms2 = self.final_conv3(hrms)


        return hrms1, hrms2