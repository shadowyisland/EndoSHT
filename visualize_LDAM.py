import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.transforms as transforms
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depthnet.networks.LDAM import LDAM

def get_attention_maps(ldam_module, f_enc, f_dec):
    """
    提取LDAM模块的各层注意力权重
    """
    attention_data = {}
    
    B, C_e, H, W = f_enc.shape
    N = H * W
    
    x_cat = torch.cat([f_enc, f_dec], dim=1)
    x = x_cat.permute(0, 2, 3, 1).reshape(B, N, -1)
    x = ldam_module.fusion_conv(x)
    
    attention_data['fused_features'] = x.detach().cpu().numpy()[0]
    
    x_res = x.transpose(1, 2).reshape(B, -1, H, W)
    local_feat = ldam_module.local_dwc(x_res)
    local_diff = x_res - local_feat
    attention_data['local_differential'] = local_diff.detach().cpu().numpy()[0]
    
    c = x.shape[-1]
    num_heads = ldam_module.num_heads
    head_dim = c // num_heads
    
    qkv = ldam_module.qkv(x).reshape(B, N, 3, c).permute(2, 0, 1, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    t_tilde = ldam_module.pool(q.reshape(B, H, W, c).permute(0, 3, 1, 2))
    t_tilde = t_tilde.reshape(B, c, -1).permute(0, 2, 1)
    
    t_pos = t_tilde + ldam_module.e_pos
    t_neg = t_tilde + ldam_module.e_neg
    
    q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    t_pos = t_pos.reshape(B, ldam_module.n, num_heads, head_dim).permute(0, 2, 1, 3)
    t_neg = t_neg.reshape(B, ldam_module.n, num_heads, head_dim).permute(0, 2, 1, 3)
    
    l_exp1 = torch.exp(torch.sum(ldam_module.lambda_q1 * ldam_module.lambda_k1, dim=-1))
    l_exp2 = torch.exp(torch.sum(ldam_module.lambda_q2 * ldam_module.lambda_k2, dim=-1))
    lambda_val = l_exp1 - l_exp2 + ldam_module.lambda_init
    
    attn_pos = (q * ldam_module.scale) @ t_pos.transpose(-2, -1)
    attn_neg = (q * ldam_module.scale) @ t_neg.transpose(-2, -1)
    attn = F.softmax(attn_pos, dim=-1) - lambda_val * F.softmax(attn_neg, dim=-1)
    
    attention_data['global_attention'] = attn.detach().cpu().numpy()[0]
    attention_data['lambda_value'] = lambda_val.item()
    
    v_pooled = ldam_module.pool(v.reshape(B, H, W, c).permute(0, 3, 1, 2)).reshape(B, c, -1).permute(0, 2, 1)
    v_pooled = v_pooled.reshape(B, ldam_module.n, num_heads, head_dim).permute(0, 2, 1, 3)
    
    global_out = attn @ v_pooled
    global_out = ldam_module.subln(global_out) * (1 - ldam_module.lambda_init)
    global_out = global_out.transpose(1, 2).reshape(B, N, c)
    global_out = ldam_module.proj(global_out)
    
    f_out = global_out.transpose(1, 2).reshape(B, c, H, W)
    gate_w = ldam_module.gate(f_out)
    
    attention_data['channel_gate'] = gate_w.detach().cpu().numpy()[0]
    
    return attention_data


def visualize_attention(image_path, output_path='attention_output.png', enc_dim=64, dec_dim=64, out_dim=64):
    """
    可视化LDAM注意力图
    
    Args:
        image_path: 输入图像路径
        output_path: 输出保存路径
        enc_dim: 编码器特征维度
        dec_dim: 解码器特征维度  
        out_dim: 输出特征维度
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ldam = LDAM(enc_dim=enc_dim, dec_dim=dec_dim, out_dim=out_dim, n=49, block_depth=2, num_heads=8).to(device)
    ldam.eval()
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    B, C, H, W = img_tensor.shape
    
    with torch.no_grad():
        f_enc = torch.randn(B, enc_dim, H, W).to(device)
        f_dec = torch.randn(B, dec_dim, H, W).to(device)
        
        f_enc = f_enc * torch.std(img_tensor, dim=1, keepdim=True).expand_as(f_enc)
        
        attention_data = get_attention_maps(ldam, f_enc, f_dec)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('LDAM Attention Visualization\n(Local Differential Attention Fusion)', fontsize=14, fontweight='bold')
    
    img_np = np.array(image.resize((H, W)))
    
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    local_diff = attention_data['local_differential']
    local_energy = np.mean(np.abs(local_diff), axis=0)
    local_normalized = (local_energy - local_energy.min()) / (local_energy.max() - local_energy.min() + 1e-8)
    axes[0, 1].imshow(local_normalized, cmap='jet')
    axes[0, 1].set_title(f'Local Differential Response\n(Lambda init={attention_data["lambda_value"]:.3f})')
    axes[0, 1].axis('off')
    
    global_attn = attention_data['global_attention']
    num_heads = global_attn.shape[0]
    head_idx = min(4, num_heads - 1)
    attn_map = np.mean(global_attn[:4], axis=0)
    attn_map_resized = cv2.resize(attn_map, (W, H))
    axes[0, 2].imshow(attn_map_resized, cmap='jet')
    axes[0, 2].set_title(f'Global Differential Attention\n(Avg over {min(4, num_heads)} heads)')
    axes[0, 2].axis('off')
    
    gate_w = attention_data['channel_gate']
    gate_energy = np.mean(gate_w, axis=0)
    gate_normalized = (gate_energy - gate_energy.min()) / (gate_energy.max() - gate_energy.min() + 1e-8)
    gate_resized = cv2.resize(gate_normalized, (W, H))
    axes[1, 0].imshow(gate_resized, cmap='hot')
    axes[1, 0].set_title('Channel Gating Weights')
    axes[1, 0].axis('off')
    
    fused_feat = attention_data['fused_features']
    feat_energy = np.mean(np.abs(fused_feat), axis=-1)
    feat_normalized = (feat_energy - feat_energy.min()) / (feat_energy.max() - feat_energy.min() + 1e-8)
    feat_resized = cv2.resize(feat_normalized, (W, H))
    axes[1, 1].imshow(feat_resized, cmap='viridis')
    axes[1, 1].set_title('Fused Features Energy')
    axes[1, 1].axis('off')
    
    combined = 0.4 * local_normalized + 0.4 * attn_map_resized + 0.2 * gate_resized
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    axes[1, 2].imshow(combined, cmap='jet')
    axes[1, 2].set_title('Combined Attention Map')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attention visualization saved to: {output_path}")
    
    return attention_data


def visualize_attention_heatmap(image_path, output_path='attention_heatmap.png', enc_dim=64, dec_dim=64, out_dim=64):
    """
    生成热力图形式的注意力可视化
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ldam = LDAM(enc_dim=enc_dim, dec_dim=dec_dim, out_dim=out_dim, n=49, block_depth=2, num_heads=8).to(device)
    ldam.eval()
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    B, C, H, W = img_tensor.shape
    
    with torch.no_grad():
        f_enc = torch.randn(B, enc_dim, H, W).to(device)
        f_dec = torch.randn(B, dec_dim, H, W).to(device)
        f_enc = f_enc * torch.std(img_tensor, dim=1, keepdim=True).expand_as(f_enc)
        attention_data = get_attention_maps(ldam, f_enc, f_dec)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('LDAM Differential Attention Mechanism - Heatmap Visualization', fontsize=14, fontweight='bold')
    
    img_np = np.array(image.resize((H, W)))
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    local_diff = attention_data['local_differential']
    local_energy = np.mean(np.abs(local_diff), axis=0)
    local_normalized = (local_energy - local_energy.min()) / (local_energy.max() - local_energy.min() + 1e-8)
    im1 = axes[1].imshow(local_normalized, cmap='hot')
    axes[1].set_title('Local Differential\n(Edge/Texture Detection)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    global_attn = attention_data['global_attention']
    attn_map = np.mean(global_attn, axis=0)
    attn_map_resized = cv2.resize(attn_map, (W, H))
    im2 = axes[2].imshow(attn_map_resized, cmap='hot')
    axes[2].set_title('Global Differential Attention\n(Semantic Regions)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    combined = 0.5 * local_normalized + 0.5 * attn_map_resized
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    im3 = axes[3].imshow(combined, cmap='hot')
    axes[3].set_title('Fused Attention\n(Local + Global)')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap visualization saved to: {output_path}")
    return attention_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize LDAM Attention')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='Input image path')
    parser.add_argument('--output', type=str, default='attention_output.png', help='Output image path')
    parser.add_argument('--enc_dim', type=int, default=64, help='Encoder feature dimension')
    parser.add_argument('--dec_dim', type=int, default=64, help='Decoder feature dimension')
    parser.add_argument('--out_dim', type=int, default=64, help='Output feature dimension')
    parser.add_argument('--heatmap', action='store_true', help='Generate heatmap visualization')
    
    args = parser.parse_args()
    
    if args.heatmap:
        visualize_attention_heatmap(args.image, args.output, args.enc_dim, args.dec_dim, args.out_dim)
    else:
        visualize_attention(args.image, args.output, args.enc_dim, args.dec_dim, args.out_dim)