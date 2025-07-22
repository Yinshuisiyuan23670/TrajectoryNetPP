import torch
import torch.nn as nn
import math
import torch.nn.functional as F

########################################
#           MSPEM 模块
########################################
class MSPEM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], directions=8):
        super(MSPEM, self).__init__()
        self.kernels = nn.ModuleList()
        for k in kernel_sizes:
            for _ in range(directions):  # 方向可以在此简化为采样等效方向数
                conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2, groups=in_channels, bias=False)
                conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2, groups=in_channels, bias=False)
                self.kernels.append(nn.Sequential(conv_x, conv_y))

    def forward(self, x):
        out = []
        for pair in self.kernels:
            gx = pair[0](x)
            gy = pair[1](x)
            grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
            out.append(grad_mag)
        return torch.cat(out, dim=1)
    
########################################
#           CKAM 模块
########################################
class SpatialSelfCorrelation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SpatialSelfCorrelation, self).__init__()
        reduced = in_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, reduced, 1)
        self.conv2 = nn.Conv2d(in_channels, reduced, 1)
        self.out_conv = nn.Conv2d(reduced, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.conv1(x).view(B, -1, H * W)
        k = self.conv2(x).view(B, -1, H * W)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        v = x.view(B, C, -1)
        out = torch.bmm(v, attn).view(B, C, H, W)
        return self.out_conv(out)

class ChannelSelfCorrelation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelSelfCorrelation, self).__init__()
        reduced = in_channels // reduction
        self.q_proj = nn.Conv2d(in_channels, reduced, 1)
        self.k_proj = nn.Conv2d(in_channels, reduced, 1)
        self.out_proj = nn.Conv2d(reduced, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.q_proj(x).view(B, -1, C)
        k = self.k_proj(x).view(B, -1, C)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        v = x.view(B, C, -1)
        out = torch.bmm(attn, v).view(B, C, H, W)
        return self.out_proj(out)

class CKAM(nn.Module):
    def __init__(self, in_channels):
        super(CKAM, self).__init__()
        self.pconv_top = nn.Conv2d(in_channels, in_channels, 1)
        self.pconv_bottom = nn.Conv2d(in_channels, in_channels, 1)
        self.ssc = SpatialSelfCorrelation(in_channels)
        self.csc = ChannelSelfCorrelation(in_channels)
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, top_feat, bottom_feat):
        top = self.pconv_top(top_feat)
        bottom = self.pconv_bottom(bottom_feat)
        x = top + bottom
        x1, x2 = torch.chunk(x, 2, dim=1)
        ssc_out = self.ssc(x1)
        csc_out = self.csc(x2)
        out = torch.cat([ssc_out, csc_out], dim=1)
        return self.fusion(out)
    

class Conv2DBlock(nn.Module):
    """ Conv2D + BN + ReLU """
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
class Triple2DConv(nn.Module):
    """ Conv2DBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

########################################
#           TrajectoryNet++ 主体
########################################
class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            Conv2DBlock(in_dim, out_dim),
            Conv2DBlock(out_dim, out_dim)
        )

    def forward(self, x):
        return self.block(x)

class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            Conv2DBlock(in_dim, out_dim),
            Conv2DBlock(out_dim, out_dim),
            Conv2DBlock(out_dim, out_dim)
        )

    def forward(self, x):
        return self.block(x)

class TrajectoryNetPlusPlus(nn.Module):
    def __init__(self, in_dim=9, out_dim=3):
        super().__init__()
        self.down1 = Double2DConv(in_dim, 64)
        self.down2 = Double2DConv(64, 128)
        self.down3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)

        self.mspem = MSPEM(256, 16)  # MSPEM输出约为16 * 3 * 8 = 384 channels
        self.ckam1 = CKAM(256 + 384)
        self.ckam2 = CKAM(128)

        self.up1 = Triple2DConv(768, 256)
        self.up2 = Double2DConv(384, 128)
        self.up3 = Double2DConv(192, 64)

        self.predictor = nn.Conv2d(64, out_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x = F.max_pool2d(x1, 2)
        x2 = self.down2(x)
        x = F.max_pool2d(x2, 2)
        x3 = self.down3(x)
        x = F.max_pool2d(x3, 2)
        x = self.bottleneck(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)

        mspem_out = self.mspem(x3)
        x = torch.cat([x, mspem_out], dim=1)
        x = self.ckam1(x, x3)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.ckam2(x, x2)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)
        x = self.sigmoid(self.predictor(x))
        return x



    
class Conv1DBlock(nn.Module):
    """ Conv1D + LeakyReLU"""
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Double1DConv(nn.Module):
    """ Conv1DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double1DConv, self).__init__()
        self.conv_1 = Conv1DBlock(in_dim, out_dim)
        self.conv_2 = Conv1DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.buttleneck = Double1DConv(128, 256)
        self.up_1 = Conv1DBlock(384, 128)
        self.up_2 = Conv1DBlock(192, 64)
        self.up_3 = Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], dim=2)                                   # (N,   L,   3)
        x = x.permute(0, 2, 1)                                         # (N,   3,   L)
        x1 = self.down_1(x)                                            # (N,  16,   L)
        x2 = self.down_2(x1)                                           # (N,  32,   L)
        x3 = self.down_3(x2)                                           # (N,  64,   L)
        x = self.buttleneck(x3)                                        # (N,  256,  L)
        x = torch.cat([x, x3], dim=1)                                  # (N,  384,  L)
        x = self.up_1(x)                                               # (N,  128,  L)
        x = torch.cat([x, x2], dim=1)                                  # (N,  192,  L)
        x = self.up_2(x)                                               # (N,   64,  L)
        x = torch.cat([x, x1], dim=1)                                  # (N,   96,  L)
        x = self.up_3(x)                                               # (N,   32,  L)
        x = self.predictor(x)                                          # (N,   2,   L)
        x = self.sigmoid(x)                                            # (N,   2,   L)
        x = x.permute(0, 2, 1)                                         # (N,   L,   2)
        return x
