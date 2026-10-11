"""BPMP-Net reconstructed from prj3113/Net2.py. State-dict names are preserved."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .common import (
    ConvBlock,
    ResidualBlock,
    ParallelMultiscaleConv,
    SerialMultiscaleConv,
    FeatureProjection,
)
from .window_attention import CrossModalAttention
from .scwab import SCWAB
from .skafe import SKAFE


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                variance_scaling_initializer(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def variance_scaling_initializer(tensor):

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == 'fan_in':
            scale /= max(1.0, fan_in)
        elif mode == 'fan_out':
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if distribution == 'normal' or distribution == 'truncated_normal':
            stddev = math.sqrt(scale) / 0.8796256610342398
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28
    variance_scaling(tensor)
    return tensor


class MCAB(nn.Module):
    """Cross-modal attention, alternating parallel/serial convolutions and PAN detail injection."""

    def __init__(self, n_feats):
        super(MCAB, self).__init__()
        self.cross_attn = CrossModalAttention(
            n_feats,
            n_heads=4,
            head_dim=8,
            win_size=4,
            cross_module=['ms', 'pan'],
            cat_feat=['ms', 'pan'],
        )
        self.four_conv_1 = ParallelMultiscaleConv(16, 20)
        self.con_1 = FeatureProjection(116, 16, 3)
        self.five_conv_1 = SerialMultiscaleConv(16, 16)
        self.con_2 = FeatureProjection(132, 16, 3)
        self.four_conv_2 = ParallelMultiscaleConv(16, 20)
        self.con_3 = FeatureProjection(232, 16, 3)
        self.five_conv_2 = SerialMultiscaleConv(16, 16)
        self.con_4 = FeatureProjection(248, 16)
        self.conv_out1 = ConvBlock(16, 8, 3, padding=1, stride=1, activation='prelu')
        self.conv_out2 = ConvBlock(8, 8, 3, padding=1, stride=1, activation='prelu')
        init_weights(
            self.cross_attn,
            self.four_conv_1,
            self.con_1,
            self.five_conv_1,
            self.con_2,
            self.four_conv_2,
            self.con_3,
            self.five_conv_2,
            self.con_4,
            self.conv_out1,
            self.conv_out2,
        )

    def forward(self, ms, pan):
        attention_features = self.cross_attn(ms, pan)
        parallel1 = self.four_conv_1(attention_features)
        projected1 = torch.cat((attention_features, parallel1), dim=1)
        projected1 = self.con_1(projected1)
        serial1 = self.five_conv_1(projected1)
        projected2 = torch.cat((attention_features, parallel1, serial1), dim=1)
        projected2 = self.con_2(projected2)
        parallel2 = self.four_conv_2(projected2)
        projected3 = torch.cat((attention_features, parallel1, serial1, parallel2), dim=1)
        projected3 = self.con_3(projected3)
        serial2 = self.five_conv_2(projected3)
        fused_features = torch.cat((attention_features, parallel1, serial1, parallel2, serial2), dim=1)
        fused_features = self.con_4(fused_features)
        pan_details = (pan - F.interpolate(
            F.interpolate(
                pan,
                scale_factor=1 / 2,
                mode='bicubic',
                align_corners=False,
                recompute_scale_factor=True,
            ),
            scale_factor=2,
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=True,
        ))
        fused_features = self.conv_out1(fused_features)
        x = fused_features * pan_details
        x = x + ms
        x = self.conv_out2(x)
        return x


class PanBranch(nn.Module):
    """PAN/MS residual preprocessing and spatial feature extraction at two scales."""

    def __init__(self):
        super(PanBranch, self).__init__()
        self.conv_input = ConvBlock(1, 32, 3, 1, 1, bias=True, activation='prelu')
        self.RB = ResidualBlock(32)
        self.conv_pan_1 = ConvBlock(32, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext1 = SKAFE(8)
        self.conv_pan_2 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext2 = SKAFE(8)
        self.downsample = ConvBlock(8, 8, 2, 2, 0, bias=True, activation='prelu')
        self.ext3 = SKAFE(8)
        self.conv_pan_3 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext4 = SKAFE(8)
        self.prelu = nn.PReLU(init=0.5)
        self.conv_12 = ConvBlock(32, 32, 3, 1, 1, bias=True, activation='prelu')
        self.conv_22 = ConvBlock(32, 32, 3, 1, 1, bias=True, activation='prelu')
        self.conv_output2 = ConvBlock(32, 8, 3, 1, 1, bias=True, activation='prelu')
        self.conv_hp_1 = ConvBlock(8, 1, 3, 1, 1, bias=True, activation='prelu')
        self.conv_hp_2 = ConvBlock(8, 1, 3, 1, 1, bias=True, activation='prelu')
        self.pixel_shuffle = nn.Sequential(nn.Conv2d(3, 3 * 4 ** 2, kernel_size=3, padding=1), nn.PixelShuffle(4))
        self.conv_ms = ConvBlock(3, 32, 3, 1, 1, bias=True, activation='prelu')

    def forward(self, pan, ms):
        ms_up4 = F.interpolate(
            ms,
            scale_factor=4,
            mode='bilinear',
            align_corners=False,
            recompute_scale_factor=True,
        )
        ms_up4 = self.conv_ms(ms_up4)
        feature_1 = self.conv_input(pan)
        feature_1 = feature_1 - ms_up4
        feature_2 = self.RB(feature_1)
        feature_3 = feature_2 + ms_up4
        feature_3 = self.conv_pan_1(feature_3)
        pan_1 = self.ext1(feature_3)
        p4 = self.conv_pan_2(pan_1)
        pan_2 = self.ext2(p4)
        p5 = self.downsample(feature_3)
        pan_3 = self.ext3(p5)
        p6 = self.conv_pan_3(pan_3)
        pan_4 = self.ext4(p6)
        hp_pan_1 = (pan - F.interpolate(
            F.interpolate(
                pan,
                scale_factor=1 / 4,
                mode='bicubic',
                align_corners=False,
                recompute_scale_factor=True,
            ),
            scale_factor=4,
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=True,
        ))
        hp_pan_2 = (self.conv_hp_1(pan_2) - F.interpolate(
            F.interpolate(
                self.conv_hp_1(pan_2),
                scale_factor=1 / 4,
                mode='bicubic',
                align_corners=False,
                recompute_scale_factor=True,
            ),
            scale_factor=4,
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=True,
        ))
        hp_pan_3 = (self.conv_hp_2(pan_4) - F.interpolate(
            F.interpolate(
                self.conv_hp_2(pan_4),
                scale_factor=1 / 2,
                mode='bicubic',
                align_corners=False,
                recompute_scale_factor=True,
            ),
            scale_factor=2,
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=True,
        ))
        return (pan_1, pan_2, pan_3, pan_4, hp_pan_1, hp_pan_2, hp_pan_3)


class BPMPNet(nn.Module):
    """Three-branch progressive pansharpening network.

    Input: PAN (B,1,256,256), BGR MS (B,3,64,64), normalized to [-1,1].
    Output: auxiliary and final HRMS, both (B,3,256,256). The auxiliary
    head follows upsampling in the historical code; it is not a 128-pixel output."""

    def __init__(self):
        super(BPMPNet, self).__init__()
        upsample = 2
        self.pan_branch = PanBranch()
        self.block1 = MCAB(8)
        self.block2 = MCAB(8)
        self.conv_input = ConvBlock(3, 32, 3, 1, 1, bias=True, activation='prelu')
        self.subpixel_up = nn.PixelShuffle(upsample)
        self.conv_hr1 = ConvBlock(8, 3, 3, 1, 1, bias=True, activation='prelu')
        self.RB1 = ResidualBlock(8)
        self.ext1 = SKAFE(8)
        self.conv_ms_11 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext2 = SKAFE(8)
        self.RB2 = ResidualBlock(8)
        self.conv_f_11 = ConvBlock(24, 8, 3, 1, 1, bias=True, activation='prelu')
        self.msa = SCWAB(n_feats=8, n_heads=4, head_dim=8, win_size=4)
        self.conv_f_12 = ConvBlock(8, 32, 3, 1, 1, bias=True, activation='prelu')
        self.conv_ms_21 = ConvBlock(8, 32, 3, 1, 1, bias=True, activation='prelu')
        self.conv_ms_22 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.ext3 = SKAFE(8)
        self.ext4 = SKAFE(8)
        self.RB3 = ResidualBlock(8)
        self.RB4 = ResidualBlock(8)
        self.final_conv1 = ConvBlock(24, 8, 3, 1, 1, bias=True, activation='prelu')
        self.final_conv2 = ConvBlock(8, 8, 3, 1, 1, bias=True, activation='prelu')
        self.final_conv3 = ConvBlock(16, 3, 3, 1, 1, bias=True, activation='prelu')
        self.conv_pan = ConvBlock(1, 8, 3, 1, 1, bias=True, activation='prelu')

    def forward(self, pan, lrms):
        # PAN features and detail guidance at 256 and 128 pixels.
        pan_1, pan_2, pan_3, pan_4, hp_1, hp_2, hp_3 = self.pan_branch(pan, lrms)

        # Half-resolution fusion: 64-pixel MS -> 128-pixel features.
        ms_1 = self.conv_input(lrms)
        ms_half = self.subpixel_up(ms_1)
        fusion_half = self.block1(ms_half, pan_4)
        fusion_half = self.RB1(fusion_half)
        ms_11 = self.ext1(ms_half)
        guided_half = fusion_half + ms_11 + hp_3 + pan_4
        guided_half = self.RB2(guided_half)
        ms_12 = self.conv_ms_11(ms_11)
        ms_13 = self.ext2(ms_12)
        fusion_full_seed = torch.cat([guided_half, ms_13, pan_3], dim=1)
        fusion_full_seed = self.conv_f_11(fusion_full_seed)
        fusion_full_seed = self.msa(fusion_full_seed)
        fusion_full_seed = self.conv_f_12(fusion_full_seed)

        # Full-resolution fusion and the historical 256-pixel auxiliary head.
        fusion_full_seed = self.subpixel_up(fusion_full_seed)
        auxiliary_hrms = self.conv_hr1(fusion_full_seed)
        fusion_full = self.block1(fusion_full_seed, pan_2)
        ms_2 = self.conv_ms_21(ms_half)
        ms_2 = self.subpixel_up(ms_2)
        ms_21 = self.ext3(ms_2)
        guided_full = fusion_full + ms_21 + hp_2 + pan_2
        guided_full = self.RB3(guided_full)
        ms_22 = self.conv_ms_22(ms_21)
        ms_23 = self.ext4(ms_22)
        full_concat = torch.cat([guided_full, ms_23, pan_1], dim=1)
        full_refined = self.final_conv1(full_concat)
        full_refined = self.msa(full_refined) + hp_1
        full_refined = self.final_conv2(full_refined)

        # Recombine the early and refined full-resolution fusion paths.
        pan = self.conv_pan(pan)
        fusion_1 = self.block2(fusion_full_seed, pan)
        fusion_2 = self.block2(full_refined, pan)
        hrms = torch.cat([fusion_1, fusion_2], dim=1)
        final_hrms = self.final_conv3(hrms)
        return (auxiliary_hrms, final_hrms)
