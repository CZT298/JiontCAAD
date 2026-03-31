import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.ndimage import morphology
from model.vit_3d import ViT
from scipy.ndimage import sobel
from thop import profile
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

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
class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm3d(width)

        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm3d(width)

        self.conv3 = nn.Conv3d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x) 
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x

# 编码部分
class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        # CNN
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        # self.vit_img_dim = img_dim // patch_dim
        self.vit_img_dim = [int(x // patch_dim) for x in img_dim]
        # print(111)
        # print(self.vit_img_dim)
        # self.vit_img_dim = img_dim // patch_dim
        # self.vit_img_dim = (16, 16, 16)
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        
        self.conv2 = nn.Conv3d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        # print(x4.shape)
        x5 = self.vit(x4)
        # print(x5.shape)
        x6 = rearrange(x5, 'b (x y z) d -> b d x y z',
                                           x=self.vit_img_dim[0], y=self.vit_img_dim[1], z=self.vit_img_dim[2])
        # x6 = rearrange(x5, "b (x y z) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        # print(x6.shape)
        x7 = self.conv2(x6)
        x8 = self.norm2(x7)
        x9 = self.relu(x8)

        return x9, x1, x2, x3

# 解码
class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv3d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        x= nn.Softmax(dim=1)(x)
        return x





class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                              head_num, mlp_dim, block_num, patch_dim)#.to('cuda:0')
        self.decoder = Decoder(out_channels, class_num)#.to('cuda:1')



    def forward(self, x):
        # x = x.to('cuda:0')
        x, x1, x2, x3 = self.encoder(x)
        #x, x1, x2, x3 = x.to('cuda:1'), x1.to('cuda:1'), x2.to('cuda:1'), x3.to('cuda:1')
        x = self.decoder(x, x1, x2, x3)
        
        return x


if __name__ == '__main__':
    import torch
    import torchsummary
    import torch.distributed as dist
    torch.cuda.empty_cache()
    # import argparse
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'
    # dist.init_process_group(backend='ncvv',init_method='env://')

    # rank = dist.get_rank() 
    # local_rank = 0
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    
    model = TransUNet(img_dim=(128, 128, 128),
                          in_channels=1,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)
    # res = model(torch.randn(1, 1, 128, 128, 128))
    from thop import profile, clever_format
    input = torch.rand(1, 1, 128,128,128)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))
    # lab = torch.randn(1, 1, 256, 256, 128).to('cuda:1')
    
    # print(transunet) 

    # dummy_input = torch.randn(1, 1, 80, 288, 288)
    # print('---------111111111111----------')
    # flops, params = profile(transunet, (dummy_input,))
    # print('params:%.2fMB' % (params / 1024 / 1024))

    # print(transunet(torch.randn(1, 1, 80, 288, 288)))
