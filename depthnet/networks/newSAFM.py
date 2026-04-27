import torch
import torch.nn as nn
import torch.nn.functional as F



class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.feature_conv(x)
        gates = self.sigmoid(self.gate_conv(x))
        return features * gates




class DynamicDepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # 使用全局平均池化获得通道描述
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 生成动态卷积核的全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels * kernel_size * kernel_size, bias=False)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        # 生成每个通道对应的动态卷积核（每个通道独立）
        context = self.global_pool(x).view(b, c)
        dynamic_kernels = self.fc(context)  # shape: [b, c * k*k]
        dynamic_kernels = dynamic_kernels.view(b, c, self.kernel_size, self.kernel_size)
        # 对每个样本、每个通道单独进行卷积操作
        out = []
        for i in range(b):
            # 使用 groups=c 实现 depthwise 卷积
            out.append(F.conv2d(x[i:i+1], dynamic_kernels[i], padding=self.padding, groups=c))
        out = torch.cat(out, dim=0)
        return out





class MultiPoolingFusionSAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        
        # 对于每个尺度，构造两个分支池化后的卷积处理模块
        self.max_conv = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(self.n_levels)]
        )
        self.avg_conv = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(self.n_levels)]
        )

        self.dynamic_convs = nn.ModuleList(
            [DynamicDepthwiseConv2d(chunk_dim, kernel_size=3) for _ in range(self.n_levels)]
        )

        self.gated_convs = nn.ModuleList(
            [GatedConv2d(chunk_dim, chunk_dim, kernel_size=3, padding=1) for _ in range(n_levels)]
        )
        # 用1×1卷积融合拼接后两倍通道的特征
        self.fuse_conv = nn.ModuleList(
            [nn.Conv2d(2 * chunk_dim, chunk_dim, 1, 1, 0) for _ in range(self.n_levels)]
        )
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        # 将特征在通道上切分为 n_levels 个部分
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // (2 ** i), w // (2 ** i))
                # 分别计算最大池化和平均池化
                max_pool = F.adaptive_max_pool2d(xc[i], p_size)
                avg_pool = F.adaptive_avg_pool2d(xc[i], p_size)
                # 分别经过各自的卷积
                max_feat = self.gated_convs[i](max_pool)
                avg_feat = self.gated_convs[i](avg_pool)
                # 拼接融合
                fused = torch.cat([max_feat, avg_feat], dim=1)
                fused = self.fuse_conv[i](fused)
                # 上采样恢复到原始尺寸
                s = F.interpolate(fused, size=(h, w), mode='nearest')
            else:
                # 第一尺度直接卷积处理，不做池化融合
                s = self.gated_convs[i](xc[i])
            out.append(s)
        # 聚合各尺度特征
        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

if __name__ == '__main__':
    input_tensor = torch.randn(3, 36, 64, 64)  # b, c, h, w
    model = MultiPoolingFusionSAFM(dim=36, n_levels=4)
    output = model(input_tensor)
    print(output.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量：{total_params / 1e6:.2f} M")  # 以百万 (M) 为单位
