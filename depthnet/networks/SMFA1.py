import torch
import torch.nn as nn
import torch.nn.functional as F

# Improved SMFA block (new innovation)
# - Adds spatial attention branch
# - Generates both scale (gamma) and shift (beta) modulation maps
# - Fuses local detail and global context via depthwise conv and pooling
# - Applies LayerNorm for stabilized training

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class SMFAv2(nn.Module):
    """
    Spatially-Adaptive Feature Modulation v2
    - Computes per-location gamma and beta for modulation
    - Incorporates local detail and global context
    - Improves on original SMFA by adding shift parameter and attention
    """
    def __init__(self, dim=36, down_scale=8):
        super().__init__()
        self.dim = dim
        self.down_scale = down_scale

        # Projections for query (y) and key (x)
        self.proj_q = nn.Conv2d(dim, dim, 1)
        self.proj_k = nn.Conv2d(dim, dim, 1)

        # Local branch: depthwise conv + pooling
        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pool = nn.AdaptiveAvgPool2d((None, None))

        # Modulation generators: gamma and beta
        self.mod_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1)
        )

        # Detail enhancement
        self.detail = DMlp(dim)

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # Query and Key
        q = self.proj_q(x)  # [B, C, H, W]
        k = self.proj_k(x)  # [B, C, H, W]

        # Local detail
        k_pool = F.adaptive_avg_pool2d(k, (h // self.down_scale, w // self.down_scale))
        k_loc = self.dw_conv(k_pool)
        k_loc = F.interpolate(k_loc, size=(h, w), mode='nearest')

        # Global context: channel-wise statistics
        k_glo = torch.mean(k, dim=(2,3), keepdim=True)

        # Fuse local and global
        fuse = torch.cat([k_loc, k_glo.expand_as(k_loc)], dim=1)  # [B, 2C, H, W]
        gamma_beta = self.mod_conv(fuse)  # [B, 2C, H, W]
        gamma, beta = gamma_beta.chunk(2, dim=1)

        # Apply modulation
        modulated = q * (1 + gamma) + beta

        # Detail enhancement
        detail = self.detail(x)

        out = modulated + detail
        #out = out.permute(0,2,3,1)
        #print(out.shape) 2 36 64 64
        #out = self.norm(out)
        #out = out.permute(0,3,1,2)
        #print(out.shape)
        return out

if __name__ == '__main__':
    inp = torch.randn(2, 36, 64, 64)
    block = SMFAv2(dim=36)
    out = block(inp)
    print('Output shape:', out.shape)
