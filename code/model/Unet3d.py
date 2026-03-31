# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:13:27 2020

@author: 陈健宇
"""

import torch.nn as nn
import torch
from torch import autograd


# 下采样和上采样路径中的连续两次卷积计算，输入、输出通道数分别为in_ch, out_ch
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 3D Single U-Net
class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        # 调整size控制网络大小
        size = 24

        # <editor-fold desc="下采样路径">
        # 连续两次卷积计算， 输入、输出通道分别为in_ch和 1 * size
        self.conv1 = DoubleConv(in_ch, 1 * size)
        # 最大池化，把图像尺寸（宽度、高度、深度）缩小一半
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = DoubleConv(1 * size, 2 * size)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DoubleConv(2 * size, 4 * size)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = DoubleConv(4 * size, 8 * size)
        self.pool4 = nn.MaxPool3d(2)
        self.conv5 = DoubleConv(8 * size, 16 * size)
        # </editor-fold>

        # <editor-fold desc="上采样路径代码块">
        # 转置卷积，使图像尺寸增大为两倍
        self.up6 = nn.ConvTranspose3d(16 * size, 8 * size, 2, stride=2)
        self.conv6 = DoubleConv(16 * size, 8 * size)
        self.up7 = nn.ConvTranspose3d(8 * size, 4 * size, 2, stride=2)
        self.conv7 = DoubleConv(8 * size, 4 * size)
        self.up8 = nn.ConvTranspose3d(4 * size, 2 * size, 2, stride=2)
        self.conv8 = DoubleConv(4 * size, 2 * size)
        self.up9 = nn.ConvTranspose3d(2 * size, 1 * size, 2, stride=2)
        self.conv9 = DoubleConv(2 * size, 1 * size)
        self.conv10 = nn.Conv3d(1 * size, out_ch, 1)
        # </editor-fold>

        # 用于Double U-Net计算的全连接层、ROI尺寸，本次未使用
        self.fnn = nn.Linear(2 * size * 2 * size * 32, 2)
        self.len_of_RIO = 96

    def unet_cnn(self, x):
        # <editor-fold desc="下采样">
        c1 = self.conv1(x)
        # print(c1.shape)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        # </editor-fold>
        # print(c5.shape)
        # <editor-fold desc="上采样">
        up_6 = self.up6(c5)
        
        # U-Net中的跳跃连接，将上采样结果up_6与下采样结果的c4在图片通道的维度叠加
        # 结果使得通道数倍增，对应下采样过程中通道数的减半
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        # print(up_6.shape,merge6.shape,c6.shape)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # 将计算结果的每个像素值通过激活函数化成(0~1)区间，得到像素的二分类预测值
        # print(c10.shape)
        #out = nn.Sigmoid()(c10)  #
        out = nn.Softmax()(c10)
        # </editor-fold>
        return out

    # 重写forward函数
    def forward(self, x):
        out0 = self.unet_cnn(x)
        return out0

if __name__ == '__main__':
    from torchsummary import summary
    
    image = torch.rand((1, 1, 128,128,128))
    model = UNet(1,1)
    # mask = model(image)
    # print(mask.shape)
    # summary(unet,(1,128,128,128),device='cpu')
    from thop import profile, clever_format
    input = torch.rand(1, 1, 128,128,128)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))