# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:13:27 2020

@author: 陈健宇
"""

import torch.nn as nn
import torch
from torch import autograd
from torchvision import transforms
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

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


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        size = 32
        self.conv1 = DoubleConv(in_ch, 1 * size)
        self.pool1 = nn.MaxPool3d(2)#.to('cuda:0')
        self.conv2 = DoubleConv(1 * size, 2 * size)#.to('cuda:0')
        self.pool2 = nn.MaxPool3d(2)#.to('cuda:0')
        self.conv3 = DoubleConv(2 * size, 4 * size)#.to('cuda:0')
        self.pool3 = nn.MaxPool3d(2)#.to('cuda:0')
        self.conv4 = DoubleConv(4 * size, 8 * size)#.to('cuda:0')
        self.pool4 = nn.MaxPool3d(2)#to('cuda:0')
        self.conv5 = DoubleConv(8 * size, 16 * size)#.to('cuda:0')
        # 逆卷积
        self.up6 = nn.ConvTranspose3d(16 * size, 8 * size, 2, stride=2)#.to('cuda:0')
        self.conv6 = DoubleConv(16 * size, 8 * size)#.to('cuda:0')
        self.up7 = nn.ConvTranspose3d(8 * size, 4 * size, 2, stride=2)#.to('cuda:0')
        self.conv7 = DoubleConv(12 * size, 4 * size)#.to('cuda:0')
        self.up8 = nn.ConvTranspose3d(4 * size, 2 * size, 2, stride=2)#.to('cuda:0')
        self.conv8 = DoubleConv(8 * size, 2 * size)#.to('cuda:0')
        self.up9 = nn.ConvTranspose3d(2 * size, 1 * size, 2, stride=2)#.to('cuda:0')
        self.conv9 = DoubleConv(5 * size, 1 * size)#.to('cuda:1')
        self.conv10 = nn.Conv3d(1 * size, out_ch, 1)#.to('cuda:1')

    def unet_cnn(self, x):
       
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)
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
        out = nn.Softmax()(c10)  # 化成(0~1)区间
        return out

    def unet(self, x):
        # <editor-fold desc="下采样">
        # x = x.to('cuda:0')
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)

        # 一层U-Net的上采样
        c1_2 = self.up9(c2)

        p2 = self.pool2(c2)
        c3 = self.conv3(p2)

        # 二层U-Net的上采样
        c2_3 = self.up8(c3)
        c1_3 = self.up9(c2_3)

        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        # 三层U-Net的上采样
        c3_4 = self.up7(c4)
        c2_4 = self.up8(c3_4)
        c1_4 = self.up9(c2_4)

        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # 四层U-Net的上采样
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        # 通过跳跃连接通道，拼接多个上采样结果
        merge7 = torch.cat([up_7, c3, c3_4], dim=1)
        c7 = self.conv7(merge7)  
        up_8 = self.up8(c7)
        # 通过跳跃连接通道，拼接多个上采样结果
        merge8 = torch.cat([up_8, c2, c2_3, c2_4], dim=1)
        
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        # 通过跳跃连接通道，拼接多个上采样结果
        merge9 = torch.cat([up_9, c1, c1_2, c1_3, c1_4], dim=1)
        merge9 = merge9#.to('cuda:1')
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Softmax()(c10)  # 化成(0~1)区间
        return out

    def forward(self, x):
        out = self.unet(x)
        return out

if __name__ == '__main__':
    # image = torch.rand((1, 1, 256,256,128))#.to(device)
    model = UNet(1,1)#.to(device)
    # mask = model(image)
    # print(mask.shape)
    from thop import profile, clever_format
    input = torch.rand(1, 1, 128,128,128)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))
    