from base_net import *
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict

# ----------------------------
# 通道注意力模块 (Channel Attention Module, CAM)
# ----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        """
        :param in_planes: 输入通道数
        :param ratio: 通道压缩比，一般取16
        """
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP：两层卷积（1x1卷积等价于全连接层）
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 分别对平均池化和最大池化的结果进行 MLP 映射
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        # 将两个结果相加后经过 Sigmoid 得到最终的通道注意力权重
        out = self.sigmoid(avg_out + max_out)
        return out


# ----------------------------
# 空间注意力模块 (Spatial Attention Module, SAM)
# ----------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7):
        """
        :param kernel_size: 卷积核大小，通常选用7
        """
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2  # 保持尺寸不变
        self.conv = nn.Conv2d(2, out_ch, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上进行平均和最大池化 -> 得到两个单通道特征图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后输入卷积
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


# ----------------------------
# CBAM 注意力模块 (结合通道和空间注意力)
# ----------------------------
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=2, kernel_size=7):
        """
        :param in_planes: 输入通道数
        :param ratio: 通道注意力的压缩比
        :param kernel_size: 空间注意力模块的卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先应用通道注意力：将输入特征按通道加权
        out = self.channel_attention(x) * x
        # 再应用空间注意力：对输出特征在空间位置上加权
        out = self.spatial_attention(out) * out
        return out




class MSKAattention(nn.Module):

    def __init__(self, channel, kernels=[1, 3, 5]):
        super(MSKAattention, self).__init__()

        # Adaptive
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2)),  # H_out=H_in
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )

        # SA
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // 2, 1, 1))  # (1, 4, 1, 1)
        self.cbias = Parameter(torch.ones(1, channel // 2, 1, 1))  # (1, 4, 1, 1)
        self.sweight = Parameter(torch.zeros(1, channel // 2, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // 2, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // 2, channel // 2)

        # utils
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        fuss = []
        # split
        for conv in self.convs:
            conv_outs.append(conv(x))
        # fuse
        U = sum(conv_outs)  # bs,c,h,w

        b, c, h, w = U.shape

        x = x.reshape(b, -1, h, w)
        assert (x.shape[1] > 1) and (x.shape[1] % 2 == 0)
        x_0, x_1 = x.chunk(2, dim=1)

        xn = self.avg_pool(x_0)  # (64, 4, 1, 1)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        return out


class PanExt(nn.Module):
    def __init__(self, channels):
        super(PanExt, self).__init__()
        self.conv_input = ConvBlock(channels, channels, 1, 1, 0, bias=True, activation='prelu')
        self.conv_11 = ConvBlock(channels, channels, 3, 1, 1, bias=True, activation='prelu')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_12 = ConvBlock(channels, channels, 1, 1, 0, bias=True, activation='prelu')

        self.conv_21 = ConvBlock(channels, channels, 3, 1, 1, bias=True, activation='prelu')
        self.conv_22 = ConvBlock(channels, channels, 5, 1, 2, bias=True, activation='prelu')
        self.conv_23 = ConvBlock(channels, channels, 7, 1, 3, bias=True, activation='prelu')
        self.conv_24 = ConvBlock(channels, channels, 9, 1, 4, bias=True, activation='prelu')
        self.conv_25 = ConvBlock(channels, channels, 13, 1, 6, bias=True, activation='prelu')

        self.sa = SpatialAttention(8, 8)
        self.ca = ChannelAttention(8)
        self.sa2 = SpatialAttention(16, 16)
        self.ca2 = ChannelAttention(16)

        self.SA1 = MSKAattention(8)
        self.SA2 = MSKAattention(16)


        self.final_1 = ConvBlock(channels, channels, 1, 1, 0, bias=True, activation='prelu')
        self.final_2 = ConvBlock(channels, channels, 3, 1, 1, bias=True, activation='prelu', norm='batch')
        self.conv = ConvBlock(channels * 2, channels, 3, 1, 1, bias=True, activation='prelu')
    def forward(self, x):
        x1 = self.conv_input(x)
        x11 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x11 = self.conv_11(x11)
        x11 = self.pool(x11)
        x1 = x11 + x1
        x1 = self.conv_12(x1)

        x21 = self.conv_21(x)
        x22 = self.conv_22(x)
        x23 = self.conv_23(x)
        x24 = self.conv_24(x)
        x25 = self.conv_25(x)

        x31 = self.SA1(x21)
        x31 = x31 + x21

        x32 = torch.cat((x22, x31), dim=1)
        x32 = self.SA2(x32)
        x32 = self.conv(x32)
        x32 = x32 + x22

        x33 = torch.cat((x23, x32), dim=1)
        x33 = self.SA2(x33)
        x33 = self.conv(x33)
        x33 = x33 + x23

        x34 = torch.cat((x24, x33), dim=1)
        x34 = self.SA2(x34)
        x34 = self.conv(x34)
        x34 = x34 + x24

        x35 = torch.cat((x25, x34), dim=1)
        x35 = self.SA2(x35)
        x35 = self.conv(x35)
        x35 = x35 + x25

        x41 = self.final_1(x31)
        x42 = self.final_1(x32)
        x43 = self.final_1(x33)
        x44 = self.final_1(x34)
        x45 = self.final_1(x35)

        # final = x41 + x42 + x43 + x44 + x45
        final = x41 + x42 + x43 + x44
        final = self.final_2(final)

        # x2 = self.conv_21(x)
        # x2 = self.conv_22(x2)
        # x2 = self.conv_23(x2)
        final = x1 * final
        return final
