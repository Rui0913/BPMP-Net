"""Shuffle selective kernel attention feature extraction (SKAFE)."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
from .common import ConvBlock


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=2):
        """
        :param in_planes: 输入通道数
        :param ratio: 通道压缩比，一般取16
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=7):
        """
        :param kernel_size: 卷积核大小，通常选用7
        """
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, out_ch, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class ShuffleSelectiveKernelAttention(nn.Module):
    """Retain the historical channel/spatial split and final channel shuffle.

    The multiscale convolution sum determines shape but is not fed to the gates in
    prj3113. Its execution is preserved, including BatchNorm running-stat updates."""

    def __init__(self, channel, kernels=[1, 3, 5]):
        super(ShuffleSelectiveKernelAttention, self).__init__()
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2)),
                            ('bn', nn.BatchNorm2d(channel)),
                            ('relu', nn.ReLU()),
                        ],
                    ),
                ),
            )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // 2, 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // 2, 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // 2, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // 2, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // 2, channel // 2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        fuss = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        U = sum(conv_outs)
        b, c, h, w = U.shape
        x = x.reshape(b, -1, h, w)
        assert x.shape[1] > 1 and x.shape[1] % 2 == 0
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out


class SKAFE(nn.Module):
    """Progressive 3/5/7/9 kernel branches with multiplicative feature recovery.

    The legacy 13-kernel path is computed but excluded from the output sum. Keep
    its modules and execution for initialization/state-dict/BatchNorm compatibility."""

    def __init__(self, channels):
        super(SKAFE, self).__init__()
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
        self.SA1 = ShuffleSelectiveKernelAttention(8)
        self.SA2 = ShuffleSelectiveKernelAttention(16)
        self.final_1 = ConvBlock(channels, channels, 1, 1, 0, bias=True, activation='prelu')
        self.final_2 = ConvBlock(channels, channels, 3, 1, 1, bias=True, activation='prelu', norm='batch')
        self.conv = ConvBlock(channels * 2, channels, 3, 1, 1, bias=True, activation='prelu')

    def forward(self, x):
        recovery = self.conv_input(x)
        recovery_up = F.interpolate(
            recovery,
            scale_factor=2,
            mode='bilinear',
            align_corners=False,
            recompute_scale_factor=True,
        )
        recovery_up = self.conv_11(recovery_up)
        recovery_up = self.pool(recovery_up)
        recovery = recovery_up + recovery
        recovery = self.conv_12(recovery)
        branch3 = self.conv_21(x)
        branch5 = self.conv_22(x)
        branch7 = self.conv_23(x)
        branch9 = self.conv_24(x)
        branch13 = self.conv_25(x)
        selected3 = self.SA1(branch3)
        selected3 = selected3 + branch3
        selected5 = torch.cat((branch5, selected3), dim=1)
        selected5 = self.SA2(selected5)
        selected5 = self.conv(selected5)
        selected5 = selected5 + branch5
        selected7 = torch.cat((branch7, selected5), dim=1)
        selected7 = self.SA2(selected7)
        selected7 = self.conv(selected7)
        selected7 = selected7 + branch7
        selected9 = torch.cat((branch9, selected7), dim=1)
        selected9 = self.SA2(selected9)
        selected9 = self.conv(selected9)
        selected9 = selected9 + branch9
        selected13 = torch.cat((branch13, selected9), dim=1)
        selected13 = self.SA2(selected13)
        selected13 = self.conv(selected13)
        selected13 = selected13 + branch13
        projected3 = self.final_1(selected3)
        projected5 = self.final_1(selected5)
        projected7 = self.final_1(selected7)
        projected9 = self.final_1(selected9)
        projected13 = self.final_1(selected13)
        final = projected3 + projected5 + projected7 + projected9
        final = self.final_2(final)
        final = recovery * final
        return final
