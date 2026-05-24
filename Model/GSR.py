import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from Model.basic import *

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# 使用HANET中的FFN
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        # hidden_features = ffn_expansion_factor
        self.project_in = nn.Conv3d(dim, hidden_features * 3, kernel_size=(1, 1, 1), bias=bias)
        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3, 3, 3),
                                 stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3),
                                 stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3),
                                 stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(1, 1, 1), bias=bias)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        x = F.gelu(x1) * x2 * x3
        x = x.unsqueeze(2)
        x = self.project_out(x)
        return x.squeeze(2)


import numbers


# 全局光谱Attention
class GlobalSpectralAttention(nn.Module):
    def __init__(self, in_channels, LayerNorm_type, num_conv_layers=3):
        super(GlobalSpectralAttention, self).__init__()
        self.num_conv_layers = num_conv_layers
        layers = []
        current_channels = in_channels
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1))
            layers.append(LayerNorm(in_channels, LayerNorm_type))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), x.size(1), -1)
        return torch.mean(x, dim=-1)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=31, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=(3, 3, 3), stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.proj(x)
        return x.squeeze(2)


class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias, num_groups=4):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3),
                                    stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3),
                                  groups=dim // self.num_heads, padding=1, bias=True)
        
        self.spectral_enhance = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        self.spectral_attention_weights = nn.Sequential(
            nn.Conv2d(1, dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, num_heads, 1),
            nn.Sigmoid()
        )
        
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        self.spectral_guidance_projection = nn.Linear(dim, dim)

    def forward(self, x, spectral):
        b, c, h, w = x.shape
        # 进行第一次校准
        spectral_weights = self.spectral_enhance(spectral)
        spectral_weights = spectral_weights.view(b, c, 1, 1)
        x = x * spectral_weights
        x_input = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x_input)).squeeze(2)
        f_all = qkv.reshape(b, h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(b, 9 * c // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv).squeeze(2)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # 获取全局光谱先验引导矩阵
        spectral_guidance = self.spectral_guidance_projection(spectral)
        spectral_guidance = rearrange(spectral_guidance, 'b (head c) -> b head c', head=self.num_heads)
        spatial_mean = x.mean(dim=[2, 3], keepdim=True)
        channel_mean = spatial_mean.mean(dim=1, keepdim=True)
        channel_mean_expanded = channel_mean.expand(b, 1, h, w)
        spectral_attention_weights = self.spectral_attention_weights(channel_mean_expanded)
        spectral_attention_weights = spectral_attention_weights.mean(dim=[2, 3]).unsqueeze(-1)
        enhanced_spectral_guidance = spectral_guidance * spectral_attention_weights
        spectral_guidance_matrix = torch.matmul(
            enhanced_spectral_guidance.unsqueeze(-1), 
            enhanced_spectral_guidance.unsqueeze(-2)
        )
        # 注意力计算
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn * spectral_guidance_matrix
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, 
            'b head c (h w) -> b (head c) h w', 
            head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out + out_conv



# class Attention1(nn.Module):
#     def __init__(self, dim, num_heads, bias, num_groups=4):
#         super(Attention1, self).__init__()
#         self.num_heads = num_heads
#         self.num_groups = num_groups
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.group_proj = nn.Conv2d(dim, num_groups, kernel_size=1, bias=bias)
#         self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
#         self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3),
#                                     stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
#         self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3),
#                                   groups=dim // self.num_heads, padding=1, bias=True)
#         self.l2q = nn.Linear(dim, dim)
#         self.spectral_diff = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim // 4, 1), nn.ReLU(),
#                                            nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid())
#     def forward(self, x, x_d):
#         b, c, h, w = x.shape
#         q_0 = self.l2q(x_d)
#         q_0 = rearrange(q_0, 'b (head c) -> b head c', head=self.num_heads)
#         q_0 = torch.matmul(q_0.unsqueeze(-1), q_0.unsqueeze(-2))
#         x = x * self.spectral_diff(x)
#         x = x.unsqueeze(2)
#         qkv = self.qkv_dwconv(self.qkv(x)).squeeze(2)
#         f_all = qkv.reshape(b, h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
#         f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)
#         f_conv = f_all.permute(0, 3, 1, 2).reshape(b, 9 * c // self.num_heads, h, w)
#         f_conv = f_conv.unsqueeze(2)
#         out_conv = self.dep_conv(f_conv).squeeze(2)
#         q, k, v = qkv.chunk(3, dim=1)
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         q = F.normalize(q, dim=-1)
#         k = F.normalize(k, dim=-1)
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn * q_0
#         attn = attn.softmax(dim=-1)
#         out = attn @ v
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         out = self.project_out(out)
#         return out + out_conv

class Attention2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention2, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3),
                                    stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3),
                                  groups=dim // self.num_heads, padding=1, bias=True)
        self.spatial_diff = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, padding=1), nn.ReLU(), nn.Conv2d(dim // 4, 1, 1),
                                          nn.Sigmoid())
    def forward(self, x):
        b, c, h, w = x.shape
        x = x * self.spatial_diff(x)
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x)).squeeze(2)
        f_all = qkv.reshape(b, h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)
        f_conv = f_all.permute(0, 3, 1, 2).reshape(b, 9 * c // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv).squeeze(2)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out + out_conv


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, avg_nums):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention1(dim, num_heads, bias)
        self.attn2 = Attention2(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.conv2avg = GlobalSpectralAttention(dim, LayerNorm_type, avg_nums)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, 2, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x, y=None):
        if y is None:
            x_d = self.conv2avg(x)
            x_d = F.gelu(x_d)
            x_att1 = self.attn1(self.norm1(x), x_d)
            x_att2 = self.attn2(self.norm1(x))
            gate = self.fusion_gate(torch.cat([x_att1, x_att2], dim=1))  #[B,2,H,W]
            x_att = gate[:, 0:1] * x_att1 + gate[:, 1:2] * x_att2  #[B,2C,H,W]
        else:
            x_d = self.conv2avg(y)
            x_d = F.gelu(x_d)
            x_att = self.attn1(self.norm1(x), x_d)
        x = x + x_att
        x = x + self.ffn(self.norm2(x))
        return x


class Block(nn.Module):
    def __init__(self, inp_channels, out_channels, dim, heads=4,
                 ffn_expansion_factor=2.66, avg_nums=3, num=2):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=False,
                LayerNorm_type='WithBias',
                avg_nums=avg_nums
            ) for _ in range(num)
        ])
        self.downsample = Down(dim, dim)
        self.bottleneck1 = TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=False, LayerNorm_type='WithBias', avg_nums=1)
        self.bottleneck2 = TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=False, LayerNorm_type='WithBias', avg_nums=1)
        self.upsample = Upm(dim, dim)
        self.decoder = nn.Sequential(*[
            TransformerBlock(
                dim=dim * 2,
                num_heads=heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=False,
                LayerNorm_type='WithBias',
                avg_nums=avg_nums
            ) for _ in range(num)
        ])
        self.output = nn.Conv3d(dim * 2, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)

    def forward(self, x, y,z):
        x_embed = self.patch_embed(x)
        y_embed = self.patch_embed(y)
        enc_input = x_embed
        enc_out = self.encoder(enc_input)
        bottleneck_in = self.downsample(enc_out)
        bottleneck_out = self.bottleneck1(self.bottleneck2(bottleneck_in, y_embed), y_embed)
        dec_in = self.upsample(bottleneck_out)
        dec_in = torch.cat([dec_in, enc_out], 1)
        dec_out = self.decoder(dec_in)
        out = self.output(dec_out.unsqueeze(2)).squeeze(2) + x
        return out
