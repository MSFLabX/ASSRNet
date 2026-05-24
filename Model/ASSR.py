import math
from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
import os
from Model.UGT import UGT
from Model.basic import *
from Model.GSR import Block
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class AdaptFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cross_attn = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels // 4, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels // 4, 2, 1),
            nn.Sigmoid()
        )
        self.feature_transform = nn.Sequential(
            FeatureTransformBlock(3 * in_channels, out_channels),
            nn.LeakyReLU(0.2)
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=1, groups=out_channels),
            nn.Conv2d(out_channels, 2 * out_channels, 1)
        )
        self.adaptive_scale = AdaptiveScale(out_channels)
        # 将非均匀分布的UGT更换
        # self.ridgelet = UGT(out_channels, num_scales=2,max_directions=8)
        self.ridgelet = UGT(out_channels, 8)
        # self.ridgelet = ConvBlock(out_channels,out_channels)
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        attn_weights = self.cross_attn(xy)
        wx, wy = attn_weights.chunk(2, dim=1)
        fused = wx * x + wy * y
        t = self.feature_transform(torch.cat([fused, xy], dim=1))
        gate_out = self.gate_conv(t)
        gate_a, gate_b = gate_out.chunk(2, dim=1)
        ridge_feat = self.ridgelet(gate_b)
        gate_weights = torch.sigmoid(ridge_feat)
        out = gate_a * gate_weights + t
        return self.adaptive_scale(out)


class DACI(nn.Module):
    def __init__(self, x_in, y_in, out_ch):
        super().__init__()
        self.conv1 = FeatureTransformBlock(x_in, x_in)
        self.conv2 = nn.Sequential(
            Down(x_in, x_in * 2),
            ConvBlock(x_in * 2, x_in * 2)
        )
        self.conv3 = nn.Sequential(
            Down(x_in, x_in * 2),
            Down(x_in * 2, x_in * 4),
            ConvBlock(x_in * 4, x_in * 4)
        )
        self.upm3 = nn.Sequential(
            Upm(x_in * 4 + 4 * y_in, x_in * 2),
            nn.Conv2d(x_in * 2, x_in * 2, 3, 1, 1, bias=True)
            # FeatureTransformBlock(x_in*2,x_in*2)
        )
        self.upm2 = nn.Sequential(
            Upm(x_in * 2 + 2 * y_in, x_in),
            nn.Conv2d(x_in, x_in, 3, 1, 1, bias=True)
            # FeatureTransformBlock(x_in,x_in)
        )
        self.y_down2 = Down(y_in, y_in * 2)
        self.y_down4 = nn.Sequential(
            Down(y_in, y_in * 2),
            Down(y_in * 2, y_in * 4)
        )
        self.conv4 = nn.Sequential(
            FeatureTransformBlock(y_in, x_in)
        )
        self.CA1 = ChannelAttention(x_in)
        self.CA2 = ChannelAttention(x_in * 2 + 2 * y_in)
        self.CA3 = ChannelAttention(x_in * 4 + 4 * y_in)
        self.out = nn.Sequential(
            FinalConv(x_in, out_ch)
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        y3 = self.y_down4(y)
        x3 = torch.cat((x3, y3), dim=1)
        x3 = self.upm3(self.CA3(x3) * x3)
        y2 = self.y_down2(y)
        x2 = torch.cat((x2 + x3, y2), dim=1)
        x2 = self.upm2(self.CA2(x2) * x2)
        x1 = self.CA1(x2 + x1) * (x2 + x1)
        out = x1 + self.conv4(y)
        return self.out(out) + self.out(x)

class DAE(nn.Module):
    def __init__(self, in_channels, h, out_channels):
        super().__init__()
        # self.ridgelet = UGT(in_channels,h)
        self.ridgelet = UGT(in_channels,4,h)
        self.low_pass = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        self.high_pass = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.Tanh()
        )
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1),  # 输入low+high=2*in_c
            nn.ReLU(),
            nn.Conv2d(in_channels, 2, 1),
            nn.Softmax(dim=1)
        )
        self.ch_attention = ChannelAttention(2 * in_channels)  # t(2)+fused(2)=2
        self.Dconv = ConvBlock(2 * in_channels, in_channels)
        self.Adapt = AdaptiveScale(out_channels)
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        t = x
        ridge_feat = self.ridgelet(t)
        low = self.low_pass(ridge_feat)
        high = self.high_pass(ridge_feat - low)
        gate = self.fusion_gate(torch.cat([low, high], dim=1))  # [B,2,H,W]
        fused = gate[:, 0:1] * low + gate[:, 1:2] * high  # [B,2C,H,W]
        combined = torch.cat([t, fused], dim=1)  # [B,2C,H,W]
        weighted = self.ch_attention(combined) * combined
        processed = self.Dconv(weighted)  # [B,C,H,W]
        x = x + processed
        return self.Adapt(self.out(x))



class ASSE(nn.Module):
    def __init__(self, x_in, y_in, h):
        super().__init__()
        self.sch1 = DACI(32, 32, 32)
        self.sch2 = DACI(32, 32, 32)
        self.tchannel1 = ChannelAttention(64)
        self.tchannel2 = ChannelAttention(64)
        
        self.xconv = DAE(x_in, h, 32)
        self.yconv = DAE(y_in, h, 32)
        self.xconv2 = DAE(32, h, 32)
        self.yconv2 = DAE(32, h, 32)
        
        self.conv3 = FeatureTransformBlock(32, 32)
        self.conv4 = FeatureTransformBlock(64, 32)
        self.conv5 = FeatureTransformBlock(64, 32)
        self.Down1 = Down(32, 32)
        self.Down2 = Down(32, 32)
        self.tDown1 = nn.Sequential(
            Down(64, 64),
            ConvBlock(64, 32)
        )
        self.tDown2 = nn.Sequential(
            Down(64, 32),
            ConvBlock(32, 32)
        )
        self.fusion1 = AdaptFusion(32, 32)
        self.fusion2 = AdaptFusion(32, 32)
        self.fusion3 = AdaptFusion(32, x_in)
        self.Conv64_32_1 = ConvBlock(64, 32)
        self.Conv64_32_2 = ConvBlock(64, 32)
        self.up1 = Upm(32, 32)
        self.up2 = Upm(32, 32)
        self.final_out = FinalConv(x_in, x_in)
    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=8, mode='bicubic', align_corners=False)  # 八倍
        x0 = x
        x = self.xconv(x) 
        x1 = x
        y = self.yconv(y)  
        y1 = y
        t1_1 = self.sch1(x, y) 
        t1_2 = self.sch1(y, x)  
        t1_c = torch.cat((t1_1, t1_2), dim=1)  
        t1_c1 = self.tDown1(self.tchannel1(t1_c) * t1_c)  
        x = self.xconv2(t1_c1 + self.Down1(x)) 
        x2 = x
        y = self.yconv2(t1_c1 + self.Down1(y))  
        y2 = y
        t2_1 = self.sch2(x, y)   
        t2_2 = self.sch2(y, x)  
        t2_c = torch.cat((t2_1, t2_2), dim=1)  
        t2_c1 = self.tDown2(self.tchannel2(t2_c) * t2_c)  
        x3 = self.conv3(t2_c1 + self.Down2(x))  
        y3 = self.conv3(t2_c1 + self.Down2(y)) 
        fu = self.Conv64_32_1(torch.cat([self.up1(self.fusion1(x3, x3+y3)), self.conv4(t2_c)], dim=1))
        fu = self.Conv64_32_2(torch.cat([self.up2(self.fusion2(fu, x2 + y2)), self.conv5(t1_c)], dim=1))
        fu = self.fusion3(fu, x1 + y1)
        return self.final_out(fu) + x0

class ASSR(nn.Module):
    def __init__(self, hsi_channels, msi_channels, h):
        super().__init__()
        self.main_branch = ASSE(hsi_channels, msi_channels, h)
        self.msi_reconstructor = nn.Sequential(
            nn.Conv2d(msi_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, hsi_channels, 1)
        )
        self.msi_projection = FeatureTransformBlock(31, 3)
        self.GSR = Block(31, 31, dim=96,ffn_expansion_factor=2.66,avg_nums=3, num=2)

    def forward(self, lr_hsi, hr_msi, is_train=False):
        hr_hsi_init = self.main_branch(lr_hsi, hr_msi)
        projected_msi = self.msi_projection(hr_hsi_init)
        msi_residual = self.msi_reconstructor(hr_msi - projected_msi)
        out = self.GSR(hr_hsi_init + msi_residual, lr_hsi,hr_msi)
        if is_train:
            return out,hr_hsi_init
        else:
            return out

if __name__ == '__main__':
    x = torch.randn(1, 31, 8, 8).cuda()
    y = torch.randn(1, 3, 64, 64).cuda()
    model = ASSR(31, 3, 64).cuda()
    t = model(x, y)
