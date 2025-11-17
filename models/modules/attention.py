
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))#不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.softmax  = Softmax(dim=-1)#对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1 ,dim:指明维度，dim=0表示按列计算；dim=1表示按行计算。默认dim的方法已经弃用了
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()#[1,3,128,128]
        proj_query = x.view(m_batchsize, C, -1)#[1,3,16384]
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)#[1,16384,3]
       
        energy = torch.bmm(proj_query, proj_key)#[1,3,16384].[1,16384,3]=[1,3,3]
        #torch.max()[0]， 只返回最大值的每个数  troch.max()[1]， 只返回最大值的每个索引

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy#[1,3,3]
        attention = self.softmax(energy_new)#[1,3,3]
        proj_value = x.view(m_batchsize, C, -1)##[1,3,16384]

        out = torch.bmm(attention, proj_value)#[1,3,3].[1,3,16384]=[1,3,16384]  == weights = torch.matmul(affinity_new, proj_value)
        out = out.view(m_batchsize, C, height, width)#[1,3,16384]==>[1,64,128,128]

        out = self.gamma*out + x
        return out

#================================================
#==============CBAM Attention====================
'''
 self.ca = ChannelAttention(planes)
 self.sa = SpatialAttention()
 out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel/reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = x * chn_se

        spa_se = self.spatial_se(x)
        spa_se = x * spa_se
        return chn_se + spa_se


class ChannelAttention1(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv=nn.Conv2d(in_planes, in_planes, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out=self.avg_pool(x)
        conv_out=self.conv(avg_out)

        return self.sigmoid(conv_out)




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)#16 is too large for remote sensing images?
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))#[1,24,128,128]==>[1,24,1,1]==>[1,24,1,1]
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))#
        avg_out=self.fc(self.avg_pool(x))#seem to work better SE-like attention
        max_out=self.fc(self.max_pool(x))
        out = avg_out + max_out
        #out = avg_out
        return self.sigmoid(out)


class ChannelAttentionHL(nn.Module):
    def __init__(self, high_ch,low_ch):
        super(ChannelAttentionHL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Conv2d(high_ch, low_ch, 1, bias=False)
        # self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)#16 is too large for remote sensing images?
        # self.relu1 = nn.ReLU()
        # self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))#[1,24,128,128]==>[1,24,1,1]==>[1,24,1,1]
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))#
        avg_out=self.fc(self.avg_pool(x))#seem to work better SE-like attention
        max_out=self.fc(self.max_pool(x))
        out = avg_out + max_out
        #out = avg_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)




if __name__ == '__main__':
    xs = torch.randn(size=(1, 32, 128, 128))
    # sa_model = semanticModule(64)  # init only input model config
    # output = sa_model(xs)  # forward input tensor
    ch_att=ChannelAttention(32)
    sp_att=SpatialAttention()
    output_ch=ch_att(xs)*xs#[1,32,1,1]*[1,32,128,128]=[1,32,128,128]
    output_sa=sp_att(xs)*xs#[1,1,128,128]*[1,32,128,128]=[1,32,128,128]
    print(output_ch.size())
    print(output_sa.size())