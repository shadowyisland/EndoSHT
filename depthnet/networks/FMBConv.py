import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution( ECCV 2024 )
# 论文地址：https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ren_The_Ninth_NTIRE_2024_Efficient_Super-Resolution_Challenge_Report_CVPRW_2024_paper.pdf

# partial convolution-based feed-forward network
class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x

class LightPCFN(nn.Module):
    def __init__(self, dim, growth_rate=1.5, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)

        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # depthwise conv 替代 3x3 conv
        self.conv_1 = nn.Conv2d(p_dim, p_dim, kernel_size=3, padding=1, groups=p_dim, bias=False)

        # 投影回原始维度
        self.conv_2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv_0(x)
        x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
        x1 = self.conv_1(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv_2(x)



class PEC_SMFA(nn.Module):
    """
    参数高效的坐标感知自调制特征聚合 (Parameter-Efficient Coordinate‑aware Self‑Modulation Feature Aggregation)
    输入:
        x: Tensor of shape (B, C, H, W)
    输出:
        modulated: Tensor of same shape, 通过空间感知的调制图 M 与原特征逐点相乘得到
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        # 深度可分离 1D 卷积：groups=channels，参数量 ≈ 3*C
        padding = kernel_size // 2
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 1) 双向全局平均池化
        #   X_h: [B, C, H, 1], X_w: [B, C, 1, W]
        x_h = F.adaptive_avg_pool2d(x, (H, 1)).view(B, C, H)
        x_w = F.adaptive_avg_pool2d(x, (1, W)).view(B, C, W)

        # 2) 共享 1D 深度可分离卷积，捕捉跨维度依赖
        #   y_h: [B, C, H], y_w: [B, C, W]
        y_h = self.conv1d(x_h)
        y_w = self.conv1d(x_w)

        # 3) 恢复空间维度并拼合注意力图
        #   y_h → [B, C, H, 1], y_w → [B, C, 1, W]
        y_h = y_h.unsqueeze(-1)
        y_w = y_w.unsqueeze(-2)

        #   M: [B, C, H, W]
        M = self.sigmoid(y_h + y_w)

        # 4) 特征自调制
        return x * M


class FMBPlusPlus(nn.Module):
    def __init__(self, dim, out_dim, ffn_scale=2.0):
        super().__init__()
        self.smfa = PEC_SMFA(dim)
        #self.pcfn = LightPCFN(dim, ffn_scale)

        # 可学习门控控制融合程度
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, out_dim, kernel_size=1)

    def forward(self, x):
        smfa_out = self.smfa(F.normalize(x))
        #pcfn_out = self.pcfn(F.normalize(smfa_out))

        out = smfa_out
        g = self.gate(x)
        o = g * out + (1 - g) * x
        o = self.proj(o)
        return o




if __name__ == '__main__':
    input_shape = (1, 36, 64, 64)
    input = torch.randn(input_shape)

    # 实例化FMB类
    block = FMB(dim=36)

    # 将输入张量传入FMB实例
    output = block(input)

    # 打印输入和输出的形状
    print(input.size())
    print(output.size())