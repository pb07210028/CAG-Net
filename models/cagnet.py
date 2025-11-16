import types
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.io import savemat

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from backbone import *


class Conv3(nn.Module):
    """
    Conv3卷积块: 3×3卷积 -> 批量归一化 -> ReLU激活
    确保输入和输出的形状保持不变
    """

    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        """
        初始化Conv3卷积块

        参数:
            in_channels: 输入特征图的通道数
            out_channels: 输出特征图的通道数
            stride: 卷积步长，默认为1
            padding: 卷积填充，默认为1（保证3×3卷积在stride=1时形状不变）
        """
        super(Conv3, self).__init__()

        # 3×3卷积层
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False  # 使用批量归一化时通常不需要偏置
        )

        # 批量归一化层
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)  # inplace=True节省内存

    def forward(self, x):
        """前向传播过程"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
# 基础卷积块
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# SPP块 (空间金字塔池化)
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks//2) for ks in kernel_sizes
        ])
        self.conv2 = ConvBlock(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        features = [x]
        for pool in self.pools:
            features.append(pool(x))
        x = torch.cat(features, dim=1)
        x = self.conv2(x)
        return x

# 上采样连接块
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, target_size):
        x = self.conv(x)
        x = self.upsample(x)
        # 确保上采样后的尺寸与目标尺寸匹配
        if x.size(2) != target_size[0] or x.size(3) != target_size[1]:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x


class MAFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDilatedConvV2, self).__init__()
        # Branch1
        self.branch1_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch1_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch1_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch1_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5)

        # Branch2
        self.branch2_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch2_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch2_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch2_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)

        # Branch3
        self.branch3_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch3_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)

        # Branch4
        self.branch4_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch4_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)

        # Branch5
        self.branch5_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 分支1前向传播
        b1 = self.branch1_1x1(x)
        b1 = self.branch1_conv1(b1)
        b1 = self.branch1_conv2(b1)
        b1 = self.branch1_conv3(b1)

        # 分支2前向传播
        b2 = self.branch2_1x1(x) + b1
        b2 = self.branch2_conv1(b2)
        b2 = self.branch2_conv2(b2)
        b2 = self.branch2_conv3(b2)

        # 分支3前向传播
        b3 = self.branch3_1x1(x) + b2
        b3 = self.branch3_conv1(b3)
        b3 = self.branch3_conv2(b3)

        # 分支4前向传播
        b4_1x1 = self.branch4_1x1(x) + b3
        b4 = self.branch4_conv1(b4_1x1)

        # 分支5前向传播（与输入残差连接）
        b5 = self.branch5_1x1(x) + b4

        return b5


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class se_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def get_module_name():
        return "se"

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, int(gate_channels // reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(gate_channels // reduction_ratio), gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=int((kernel_size - 1) // 2), relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class cbam_module(nn.Module):
    def __init__(self, gate_channels, reduction=16, pool_types=['avg', 'max'], no_spatial=False):
        super(cbam_module, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    @staticmethod
    def get_module_name():
        return "cbam"

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class CAGNet(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, depths=[3, 3, 4, 3], heads=[4, 4, 4, 4],
                 enc_channels=[64, 96, 128, 256]):
        super(CAGNet, self).__init__()

        # The data and the code of this study will be shared with the respective authors upon reasonable request.