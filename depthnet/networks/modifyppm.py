import torch
from torch import nn
from .common import LayerNorm2d
from torch.nn import functional as F


class ModifyPPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(ModifyPPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, bias=False, groups=reduction_dim),
                nn.GELU()
            ))
        self.features = nn.ModuleList(self.features)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False, groups=in_dim),
            nn.GELU(),
        )
        self.proconv = nn.Conv2d(in_dim*2,in_dim,kernel_size=1)

    def forward(self, x):
        x_size = x.size()

        out = [self.local_conv(x)]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        o = torch.cat(out, 1)
        o =self.proconv(o)
        return o
if __name__ == '__main__':
    block = ModifyPPM(64,16,[3,6,9,12])
    input_tensor = torch.rand(1, 64, 64, 64)
    output = block(input_tensor)
    print(input_tensor.size(), output.size())
    total_params = sum(p.numel() for p in block.parameters())
    print(f"模型参数总量：{total_params / 1e6:.2f} M")  # 以百万 (M) 为单位
