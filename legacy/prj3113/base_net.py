import torch
import math
import torch.nn as nn
import torch.nn.functional as F


######################################
#            common model
######################################

class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, activation='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        if scale == 3:
            modules.append(ConvBlock(n_feat, 9 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(3))
            if bn:
                modules.append(torch.nn.BatchNorm2d(n_feat))
        else:
            for _ in range(int(math.log(scale, 2))):
                modules.append(ConvBlock(n_feat, 64, 3, 1, 1, bias, activation='prelu', norm=None))
                modules.append(ConvBlock(64, 32, 3, 1, 1, bias, activation='prelu', norm=None))
                modules.append(ConvBlock(32, 4 * n_feat, 3, 1, 1, bias, activation='prelu', norm=None))
                modules.append(torch.nn.PixelShuffle(2))
                if bn:
                    modules.append(torch.nn.BatchNorm2d(n_feat))

        self.up = torch.nn.Sequential(*modules)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        if self.pad_model == None:
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding,
                                        bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))   #反射填充是指在边缘填充时使用输入数据的反射值，而不是简单的零填充或常数填充。
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,
                                        bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out



class ResnetBlock(torch.nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu',
                 norm='batch', pad_model=None):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale

        if self.norm == 'batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None,
                        [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer,
                         self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out


class four_conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3):
        super(four_conv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv0 = nn.Sequential(
            ConvBlock(self.input_size, self.output_size, kernel_size=1, padding=0, stride=1, activation='prelu'),
        )
        self.conv1 = nn.Sequential(
            ConvBlock(self.input_size, self.output_size, kernel_size=3, padding=1, stride=1, activation='prelu'),
        )
        self.conv2 = nn.Sequential(
            ConvBlock(self.input_size, self.output_size, kernel_size=5, padding=2, stride=1, activation='prelu'),
        )
        self.conv3 = nn.Sequential(
            ConvBlock(self.input_size, self.output_size, kernel_size=7, padding=3, stride=1, activation='prelu'),
        )
        self.conv4 = nn.Sequential(
            ConvBlock(self.input_size, self.output_size, kernel_size=1, padding=0, stride=1, activation='prelu'),
        )

    def forward(self, x):
        branch0 = self.conv0(x)
        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        branch3 = self.conv3(x)
        branch4 = self.conv4(x)
        output = torch.cat([branch0, branch1, branch2, branch3,branch4], dim=1)

        return output


class five_conv(nn.Module):
    def __init__(self, input_size, output_size):
        super(five_conv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv = nn.Sequential(
            ConvBlock(self.input_size, self.output_size, kernel_size=1, padding=0, stride=1, activation='prelu'),   #1
            ConvBlock(self.input_size, self.output_size, kernel_size=3, padding=1, stride=1, activation='prelu'),
            ConvBlock(self.input_size, self.output_size, kernel_size=7, padding=3, stride=1, activation='prelu'),
            ConvBlock(self.input_size, self.output_size, kernel_size=5, padding=2, stride=1, activation='prelu'),
            ConvBlock(self.input_size, self.output_size, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class convert_conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3):
        super(convert_conv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            ConvBlock(self.input_size, self.output_size, self.kernel_size, padding=1, stride=1,activation='prelu'),
            nn.BatchNorm2d(self.output_size),
            nn.PReLU(init=0.5),
            ResnetBlock(self.output_size),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class LFE(nn.Module):
    def __init__(self, channels):
        super(LFE, self).__init__()
        self.conv_input = ConvBlock(channels, channels, 1, 1, 0, bias=True, activation='prelu')
        self.conv_11 = ConvBlock(channels, channels, 3, 1, 1, bias=True, activation='prelu')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_12 = ConvBlock(channels, channels, 1, 1, 0, bias=True, activation='prelu')

        self.conv_21 = ConvBlock(channels, channels, 3, 1, 1, bias=True, activation='prelu')
        self.conv_22 = ConvBlock(channels, channels, 3, 1, 1, bias=True, activation='prelu')
        self.conv_23 = ConvBlock(channels, channels, 3, 1, 1, bias=True, activation='prelu')

    def forward(self, x):
        x1 = self.conv_input(x)
        x11 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x11 = self.conv_11(x11)
        x11 = self.pool(x11)
        x1 = x11 + x1
        x1 = self.conv_12(x1)

        x2 = self.conv_21(x)
        x2 = self.conv_22(x2)
        x2 = self.conv_23(x2)
        final = x1 * x2
        return final

