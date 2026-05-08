import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

def lambda_init_fn(depth):
    # 根据深度动态调整初始λ，深度越深，差分权重相对越稳定
    if depth is None: depth = 1
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class LDAM(nn.Module):
    """
    两级差分注意力融合模块 (Two-level Differential Attention Fusion Module)
    用于肠道内窥镜深度估计的跳跃连接处。
    """
    def __init__(self, enc_dim, dec_dim, out_dim, n=49, block_depth=1, num_heads=8):
        super().__init__()
        self.n = n  # 全局视觉令牌数量
        self.num_heads = num_heads
        
        # 1. 特征拼接与对齐
        self.fusion_conv = nn.Sequential(
            nn.Linear(enc_dim + dec_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        head_dim = out_dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 2. 局部差分交互 (Local-level)
        # 使用DW-Conv模拟局部窗口内的特征差异提取
        self.local_dwc = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim)
        self.local_norm = RMSNorm(out_dim)
        
        # 3. 全局差分注意力 (Global-level)
        self.qkv = nn.Linear(out_dim, out_dim * 3)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(int(n**0.5), int(n**0.5)))
        
        # 视觉令牌偏置 (Positive/Negative Streams)
        self.e_pos = nn.Parameter(torch.randn(1, n, out_dim) * 0.02)
        self.e_neg = nn.Parameter(torch.randn(1, n, out_dim) * 0.02)
        
        # 差分系数 λ
        self.lambda_init = lambda_init_fn(block_depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        
        self.subln = RMSNorm(head_dim)
        self.proj = nn.Linear(out_dim, out_dim)
        
        # 4. 可学习通道门控分支
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 4, out_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, f_enc, f_dec):
        """
        f_enc: 编码器特征 [B, C1, H, W]
        f_dec: 解码器上采样特征 [B, C2, H, W]
        """
        B, C_e, H, W = f_enc.shape
        N = H * W
        
        # --- 步骤 1: 特征拼接与空间展平 ---
        # [B, C1+C2, H, W]
        x_cat = torch.cat([f_enc, f_dec], dim=1)
        x = x_cat.permute(0, 2, 3, 1).reshape(B, N, -1)
        x = self.fusion_conv(x) # [B, N, out_dim]
        
        # --- 步骤 2: 局部级差分响应 (Local-level) ---
        # 模拟局部窗口对黏膜、血管边缘的捕捉
        x_res = x.transpose(1, 2).reshape(B, -1, H, W)
        local_feat = self.local_dwc(x_res)
        # 显式计算差分响应: 原始特征 - 局部平滑特征
        local_diff = x_res - local_feat 
        local_out = self.local_norm(local_diff.permute(0, 2, 3, 1).reshape(B, N, -1))
        x = x + local_out
        
        # --- 步骤 3: 全局级差分注意力 (Global-level) ---
        c = x.shape[-1]
        num_heads = self.num_heads
        head_dim = c // num_heads
        
        qkv = self.qkv(x).reshape(B, N, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 全局蒸馏生成视觉令牌 t_tilde
        # [B, out_dim, h, w] -> [B, n, out_dim]
        t_tilde = self.pool(q.reshape(B, H, W, c).permute(0, 3, 1, 2))
        t_tilde = t_tilde.reshape(B, c, -1).permute(0, 2, 1)
        
        # 构建正负流: t_+ = t_tilde + e^+, t_- = t_tilde + e^-
        t_pos = t_tilde + self.e_pos
        t_neg = t_tilde + self.e_neg
        
        # 维度变换
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        t_pos = t_pos.reshape(B, self.n, num_heads, head_dim).permute(0, 2, 1, 3)
        t_neg = t_neg.reshape(B, self.n, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # 计算全局差分系数 λ
        l_exp1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        l_exp2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_val = l_exp1 - l_exp2 + self.lambda_init
        
        # 差分交互：A = Softmax(q @ t_+^T) - λ * Softmax(q @ t_-^T)
        attn_pos = (q * self.scale) @ t_pos.transpose(-2, -1)
        attn_neg = (q * self.scale) @ t_neg.transpose(-2, -1)
        attn = F.softmax(attn_pos, dim=-1) - lambda_val * F.softmax(attn_neg, dim=-1)
        
        # 聚合全局信息
        v_pooled = self.pool(v.reshape(B, H, W, c).permute(0, 3, 1, 2)).reshape(B, c, -1).permute(0, 2, 1)
        v_pooled = v_pooled.reshape(B, self.n, num_heads, head_dim).permute(0, 2, 1, 3)
        
        global_out = attn @ v_pooled # [B, M, N, d]
        global_out = self.subln(global_out) * (1 - self.lambda_init)
        global_out = global_out.transpose(1, 2).reshape(B, N, c)
        global_out = self.proj(global_out)
        
        # --- 步骤 4: 通道门控与最终融合 ---
        f_out = global_out.transpose(1, 2).reshape(B, c, H, W)
        gate_w = self.gate(f_out)
        
        # 动态加权原始投影特征与差分注意力特征
        out = (1 - gate_w) * x_res + gate_w * f_out
        
        return out