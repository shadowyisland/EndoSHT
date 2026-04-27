import torch
from torch import nn
import torch.nn.functional as F

class ImprovedEMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(ImprovedEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        # 多种池化方式
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
        # 用膨胀卷积替换普通的3x3卷积，膨胀率可调
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=2, dilation=2)
        # 新增1x1卷积用于融合全局池化结果
        self.fuse_conv = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # [b*groups, c//groups, h, w]

        # 多尺度池化：平均池化与最大池化
        x_avg = self.agp(group_x)
        x_max = self.max_pool(group_x)
        pooled = x_avg + x_max  # [b*groups, c//groups, 1, 1]
        # 融合全局描述
        global_mod = self.fuse_conv(pooled)  # [b*groups, c//groups, 1, 1]
        # 将全局描述扩展后调制输入特征
        group_x = group_x * global_mod.expand_as(group_x)
        
        # 沿高度和宽度方向分别1D池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 引入温度参数控制 Sigmoid 的锐化程度
        x1 = self.gn(group_x * (x_h.sigmoid() * self.temperature) * (x_w.permute(0, 1, 3, 2).sigmoid() * self.temperature))
        x2 = self.conv3x3(group_x)
        
        # 计算注意力权重
        x11 = self.softmax((self.agp(x1).reshape(b * self.groups, -1, 1) / self.temperature).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax((self.agp(x2).reshape(b * self.groups, -1, 1) / self.temperature).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return out

if __name__ == '__main__':
    block = ImprovedEMA(64, factor=8)
    input_tensor = torch.rand(1, 64, 64, 64)
    output = block(input_tensor)
    print(input_tensor.size(), output.size())
    total_params = sum(p.numel() for p in block.parameters())
    print(f"模型参数总量：{total_params / 1e6:.2f} M")  # 以百万 (M) 为单位
