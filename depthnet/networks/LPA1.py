import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class LPA(nn.Module):
    def __init__(self, in_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        # 将特征图按照高度和宽度分别划分为两半，然后组合成四个象限
        h, w = x.size(2), x.size(3)
        h_half, w_half = h // 2, w // 2
        q1 = x[:, :, :h_half, :w_half]
        q2 = x[:, :, :h_half, w_half:]
        q3 = x[:, :, h_half:, :w_half]
        q4 = x[:, :, h_half:, w_half:]

        # 对每个象限分别应用通道注意力和空间注意力
        q1 = self.sa(self.ca(q1)) * q1
        q2 = self.sa(self.ca(q2)) * q2
        q3 = self.sa(self.ca(q3)) * q3
        q4 = self.sa(self.ca(q4)) * q4

        # 将四个象限还原成完整的特征图
        top = torch.cat([q1, q2], dim=3)
        bottom = torch.cat([q3, q4], dim=3)
        x3 = torch.cat([top, bottom], dim=2)

        # 同时，对整个特征图也做通道和空间注意力
        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4

        # 将局部四象限特征与全局特征进行融合
        out = x3 + x4
        return out

if __name__ == '__main__':
    input = torch.rand(12, 256, 8, 10)
    block = LPA(in_channel=256)
    output = block(input)

    print("Input size:", input.size())
    print("Output size:", output.size())
