"""Spatial-channel window self-attention block (SCWAB)."""
import torch
from torch import nn
from .common import ConvBlock
from .window_attention import SwinModule


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(combined))
        return x * spatial_weights


class SCWAB(nn.Module):
    """Four stages: spatial/window, channel/window, spatial/window, channel/window.

    Dense concatenation uses channel projections; one window module is shared by all stages."""

    def __init__(self, n_feats=8, n_heads=4, head_dim=8, win_size=4):
        super().__init__()
        self.n_blocks = 3
        self.atten = SwinModule(
            in_channels=n_feats,
            hidden_dimension=n_feats,
            layers=2,
            downscaling_factor=1,
            num_heads=n_heads,
            head_dim=head_dim,
            window_size=win_size,
            relative_pos_embedding=True,
            cross_attn=False,
        )
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(8)
        self.conv1 = ConvBlock(16, 8, 3, 1, 1, bias=True, activation='prelu')
        self.conv2 = ConvBlock(24, 8, 3, 1, 1, bias=True, activation='prelu')
        self.conv3 = ConvBlock(32, 8, 3, 1, 1, bias=True, activation='prelu')

    def forward(self, x):
        spatial1 = self.sa(x)
        spatial1 = self.atten(spatial1)
        channel1 = torch.cat([x, spatial1], dim=1)
        channel1 = self.conv1(channel1)
        channel1 = self.ca(channel1)
        channel1 = self.atten(channel1)
        spatial2 = torch.cat([x, spatial1, channel1], dim=1)
        spatial2 = self.conv2(spatial2)
        spatial2 = self.sa(spatial2)
        spatial2 = self.atten(spatial2)
        channel2 = torch.cat([x, spatial1, channel1, spatial2], dim=1)
        channel2 = self.conv3(channel2)
        channel2 = self.ca(channel2)
        channel2 = self.atten(channel2)
        final = channel2
        return final
