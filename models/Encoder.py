import torch
from thop import profile
from thop import clever_format
from fvcore.nn import FlopCountAnalysis, flop_count_table

## ElgcNet
import types
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy.io import savemat

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class OverlapPatchEmbed(nn.Module):
    """
    Patch Embedding layer to donwsample spatial resolution before each stage.
    in_chans: number of channels of input features
    embed_dim: number of channels for output features
    patch_size: kernel size of the convolution
    stride: stride value of convolution
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return x, H, W


################## Encoder Modules ####################
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        # self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        # x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class ELGCA(nn.Module):
    """
    Efficient local global context aggregation module
    dim: number of channels of input
    heads: number of heads utilized in computing attention
    """

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)
        self.qkvl = nn.Conv2d(dim // 2, (dim // 4) * self.heads, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x1, x2 = torch.split(x, [C // 2, C // 2], dim=1)
        # apply depth-wise convolution on half channels
        x1 = self.act(self.dwconv(x1))

        # linear projection of other half before computing attention
        x2 = self.act(self.qkvl(x2))

        x2 = x2.reshape(B, self.heads, C // 4, H, W)

        q = torch.sum(x2[:, :-3, :, :, :], dim=1)
        k = x2[:, -3, :, :, :]

        q = self.pool_q(q)
        k = self.pool_k(k)

        v = x2[:, -2, :, :, :].flatten(2)
        lfeat = x2[:, -1, :, :, :]

        qk = torch.matmul(q.flatten(2), k.flatten(2).transpose(1, 2))
        qk = torch.softmax(qk, dim=1).transpose(1, 2)

        x2 = torch.matmul(qk, v).reshape(B, C // 4, H, W)

        x = torch.cat([x1, lfeat, x2], dim=1)

        return x


class EncoderBlock(nn.Module):
    """
    dim: number of channels of input features
    """

    def __init__(self, dim, drop_path=0.1, mlp_ratio=4, heads=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.attn = ELGCA(dim, heads=heads)

    def forward(self, x, H, W):
        B, C, H, W = x.shape
        inp_copy = x

        x = self.layer_norm1(inp_copy)
        x = self.attn(x)
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.mlp(x)
        out = out + x
        return out

################## Encoder #########################
# Transormer Ecoder with x4, x8, x16, x32 scales
class Encoder(nn.Module):
    def __init__(self, patch_size=3, in_chans=3, num_classes=2, embed_dims=[64, 96, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], drop_path_rate=0., heads=[4, 4, 4, 4],
                 depths=[3, 3, 4, 3]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=patch_size, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        ############ Stage-1 (x1/4 scale)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # cur = 0

        self.block1 = nn.ModuleList()
        for i in range(depths[0]):
            self.block1.append(EncoderBlock(dim=embed_dims[0], mlp_ratio=mlp_ratios[0]))

        ############# Stage-2 (x1/8 scale)
        # cur += depths[0]

        self.block2 = nn.ModuleList()
        for i in range(depths[1]):
            self.block2.append(EncoderBlock(dim=embed_dims[1], mlp_ratio=mlp_ratios[1]))

        ############# Stage-3 (x1/16 scale)
        # cur += depths[1]

        self.block3 = nn.ModuleList()
        for i in range(depths[2]):
            self.block3.append(EncoderBlock(dim=embed_dims[2], mlp_ratio=mlp_ratios[2]))

        ############# Stage-4 (x1/32 scale)
        # cur += depths[2]

        self.block4 = nn.ModuleList()
        for i in range(depths[3]):
            self.block4.append(EncoderBlock(dim=embed_dims[3], mlp_ratio=mlp_ratios[3]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        outs.append(x1)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDilatedConv, self).__init__()

        # 分支1：单个3×3空洞卷积，膨胀率r=1
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)

        # 分支2：两个3×3空洞卷积，膨胀率分别为r=1、r=2
        self.branch2_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch2_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)

        # 分支3：三个3×3空洞卷积，膨胀率分别为r=1、r=1、r=3
        self.branch3_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch3_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch3_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)

        # 分支4：三个3×3空洞卷积，膨胀率分别为r=1、r=1、r=5
        self.branch4_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch4_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.branch4_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5)

        # 1×1卷积用于融合多分支特征
        self.fusion_conv = nn.Conv2d(4 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 分支1前向传播
        out1 = self.branch1(x)

        # 分支2前向传播
        out2 = self.branch2_conv1(x)
        out2 = self.branch2_conv2(out2)

        # 分支3前向传播
        out3 = self.branch3_conv1(x)
        out3 = self.branch3_conv2(out3)
        out3 = self.branch3_conv3(out3)

        # 分支4前向传播
        out4 = self.branch4_conv1(x)
        out4 = self.branch4_conv2(out4)
        out4 = self.branch4_conv3(out4)

        # 拼接所有分支的输出
        fused = torch.cat([out1, out2, out3, out4], dim=1)
        # 1×1卷积融合特征
        fused = self.fusion_conv(fused)
        # 残差连接（输入x与融合特征相加）
        out = fused + x
        return out


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



