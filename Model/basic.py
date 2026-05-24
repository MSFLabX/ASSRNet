import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = in_channels // reduction_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=1 / 2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Upm(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class FeatureTransformBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureTransformBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 2 * in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1)
        )
        self.channel = ChannelAttention(in_channels * 2)
        self.conv3 = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x1 = self.conv3(x)
        x = self.conv1(x)
        x = x * self.channel(x)
        x = self.conv2(x)
        return x + x1


# class FeatureTransformBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction_ratio=4, num_heads=1):
#         super(FeatureTransformBlock, self).__init__()
        
#         self.split_ratio = 0.5
#         mid_channels = in_channels // 2
        
#         self.main_branch = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, 1),  # 通道压缩
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels),
#             nn.Conv2d(mid_channels, mid_channels, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
        
#         self.context_branch = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, 3, padding=1, dilation=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(mid_channels, mid_channels, 3, padding=2, dilation=2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
        
#         self.light_ca = LightChannelAttention(mid_channels, reduction_ratio)
        
#         self.feature_interaction = CrossStreamInteraction(mid_channels, num_heads)
        

#         self.fusion = nn.Sequential(
#             nn.Conv2d(2 * mid_channels, out_channels, 1),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
#     def forward(self, x):
#         residual = self.residual(x)
#         main_feat = self.main_branch(x)
#         context_feat = self.context_branch(x)
#         main_feat = self.light_ca(main_feat) * main_feat
#         context_feat = self.light_ca(context_feat) * context_feat
#         main_enhanced, context_enhanced = self.feature_interaction(main_feat, context_feat)
#         fused = torch.cat([main_enhanced, context_enhanced], dim=1)
#         out = self.fusion(fused)
#         return out + residual

# class LightChannelAttention(nn.Module):
#     def __init__(self, channels, reduction_ratio=4):
#         super().__init__()
#         reduced_channels = max(4, channels // reduction_ratio)
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, reduced_channels, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(reduced_channels, channels, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         return self.attention(x)

# class CrossStreamInteraction(nn.Module):
#     def __init__(self, channels, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.channels = channels
#         self.head_dim = channels // num_heads
#         self.to_q = nn.Conv2d(channels, channels, 1)
#         self.to_k = nn.Conv2d(channels, channels, 1)
#         self.to_v = nn.Conv2d(channels, channels, 1)
#         self.gate = nn.Sequential(
#             nn.Conv2d(2 * channels, channels, 1),
#             nn.Sigmoid()
#         )
#         self.norm = nn.GroupNorm(num_heads, channels)
#     def forward(self, stream1, stream2):
#         B, C, H, W = stream1.shape

#         q1 = self.to_q(stream1).view(B, self.num_heads, self.head_dim, H * W)
#         k2 = self.to_k(stream2).view(B, self.num_heads, self.head_dim, H * W)
#         v2 = self.to_v(stream2).view(B, self.num_heads, self.head_dim, H * W)

#         attn = torch.softmax(torch.matmul(q1.transpose(2, 3), k2) / (self.head_dim ** 0.5), dim=-1)

#         stream1_enhanced = torch.matmul(v2, attn.transpose(2, 3)).view(B, C, H, W)

#         q2 = self.to_q(stream2).view(B, self.num_heads, self.head_dim, H * W)
#         k1 = self.to_k(stream1).view(B, self.num_heads, self.head_dim, H * W)
#         v1 = self.to_v(stream1).view(B, self.num_heads, self.head_dim, H * W)
        
#         attn2 = torch.softmax(torch.matmul(q2.transpose(2, 3), k1) / (self.head_dim ** 0.5), dim=-1)
#         stream2_enhanced = torch.matmul(v1, attn2.transpose(2, 3)).view(B, C, H, W)
        
#         # 门控融合
#         gate1 = self.gate(torch.cat([stream1, stream1_enhanced], dim=1))
#         gate2 = self.gate(torch.cat([stream2, stream2_enhanced], dim=1))
        
#         stream1_out = gate1 * stream1_enhanced + (1 - gate1) * stream1
#         stream2_out = gate2 * stream2_enhanced + (1 - gate2) * stream2
        
#         return self.norm(stream1_out), self.norm(stream2_out)


class SparseFeature(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sparse_weight = nn.Parameter(torch.randn(1, channels, 1, 1))
    def forward(self, x):
        sparse_weight = F.sigmoid(self.sparse_weight)
        return x * sparse_weight
        
class AdaptiveScale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, att):
        return att * self.scale



class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

def mode_n_product(tensor, matrix, mode):
    if mode == 1:
        original_dim = tensor.shape[0]
        new_dim = matrix.shape[0]
        # assert matrix.shape[1] == original_dim, f"Mode-1 矩阵列数必须为 {original_dim}，但输入为 {matrix.shape[1]}"
        tensor_reshaped = tensor.reshape(original_dim, -1) 
        # print(tensor_reshaped.shape)
        product_reshaped = matrix @ tensor_reshaped
        return product_reshaped.reshape(new_dim, tensor.shape[1], tensor.shape[2])  # (new_dim, d2, d3)
    
    elif mode == 2:
        original_dim = tensor.shape[1]
        new_dim = matrix.shape[0]
        # assert matrix.shape[1] == original_dim, f"Mode-2 矩阵列数必须为 {original_dim}，但输入为 {matrix.shape[1]}"
        tensor_reshaped = tensor.permute(1, 0, 2).reshape(original_dim, -1)  
        product_reshaped = matrix @ tensor_reshaped
        return product_reshaped.reshape(new_dim, tensor.shape[0], tensor.shape[2]).permute(1, 0, 2) 
    
    elif mode == 3:
        original_dim = tensor.shape[2]
        new_dim = matrix.shape[0]
        # assert matrix.shape[1] == original_dim, f"Mode-3 矩阵列数必须为 {original_dim}，但输入为 {matrix.shape[1]}"
        tensor_reshaped = tensor.permute(2, 0, 1).reshape(original_dim, -1) 
        product_reshaped = matrix @ tensor_reshaped 
        return product_reshaped.reshape(new_dim, tensor.shape[0], tensor.shape[1]).permute(1, 2, 0)  
    
    raise ValueError("Mode must be 1, 2, or 3")


class TuckerConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, rank_ratio=0.25):
        super().__init__()
        self.in_channels = in_channels  
        self.out_channels = out_channels  
        self.kernel_size = kernel_size 
        self.stride = stride
        self.padding = padding
        self.rank_c = max(3, int(in_channels * rank_ratio)) 
        self.rank_k = max(3, int(kernel_size * rank_ratio))  
        self.core = nn.Parameter(torch.randn(self.rank_c, self.rank_k, self.rank_k))
        self.W_in = nn.Parameter(torch.randn(in_channels, self.rank_c))
        self.W_k1 = nn.Parameter(torch.randn(kernel_size, self.rank_k))
        self.W_k2 = nn.Parameter(torch.randn(kernel_size, self.rank_k))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.core)
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_k1)
        nn.init.xavier_uniform_(self.W_k2)
    def forward(self, x):
        kernel = mode_n_product(self.core, self.W_in, mode=1)  
        kernel = mode_n_product(kernel, self.W_k1, mode=2)
        kernel = mode_n_product(kernel, self.W_k2, mode=3) 
        kernel = kernel.unsqueeze(0) 
        kernel = kernel.repeat(self.out_channels, 1, 1, 1) 
        out = F.conv2d(x, kernel, self.bias, self.stride, self.padding)
        return out



