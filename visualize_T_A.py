from __future__ import absolute_import, division, print_function

import sys
import os
import random

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


P = 32
H_PATCHES = 8
W_PATCHES = 10
PATCH_DIM = P ** 2 * 3


def patchify(imgs):
    assert imgs.shape[2] % P == 0
    h = imgs.shape[2] // P
    w = imgs.shape[3] // P
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, P, w, P))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, PATCH_DIM))
    return x


def unpatchify(x):
    assert H_PATCHES * W_PATCHES == x.shape[1]
    x = x.reshape(shape=(x.shape[0], H_PATCHES, W_PATCHES, P, P, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, H_PATCHES * P, W_PATCHES * P))
    return imgs


def deal_uncertainty_patch(patch):
    uncertainty_patch = patch
    uncertainty_mask = torch.ones((patch.shape[0], patch.shape[1]))

    for i in range(uncertainty_patch.shape[0]):
        k = int(0.75 * len(uncertainty_patch[i]))
        largest_k = torch.topk(uncertainty_patch[i], k)
        k_value = largest_k.values
        k_value = k_value[-1]
        for j in range(uncertainty_patch.shape[1]):
            if uncertainty_patch[i][j] <= k_value:
                uncertainty_mask[i][j] = 0

        for j in range(uncertainty_patch.shape[1]):
            if uncertainty_mask[i][j] == 0:
                seed = random.random() > 0.5
                if seed:
                    uncertainty_mask[i][j] = 1
            elif uncertainty_mask[i][j] == 1:
                seed = random.random() > 1 - 0.5 * 0.25 / (1 - 0.25)
                if seed:
                    uncertainty_mask[i][j] = 0

    return uncertainty_mask


def random_masking(x, mask_ratio, device='cpu'):
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def load_image(image_path, target_height=256, target_width=320):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img = img.resize((target_width, target_height), Image.LANCZOS)
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)
    return img_tensor, original_size


def save_tensor_image(tensor, path):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_mask_image(mask, h_patches, w_patches, patch_size, path):
    mask_map = mask.reshape(1, h_patches, w_patches).float()
    mask_map = mask_map.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    mask_map = mask_map.squeeze(0).cpu().numpy()
    mask_map = (mask_map * 255).astype(np.uint8)
    Image.fromarray(mask_map).save(path)


def main():
    # ==================== 硬编码配置（按需修改） ====================
    IMAGE_PATH = r"D:\your_image.jpg"
    OUTPUT_DIR = r"D:\output"
    MASK_RATIO = 0.75
    DEVICE = "cpu"
    # 注意：尺寸必须固定为 256x320，因为 T_A.py 的 unpatchify 硬编码了 h=8, w=10
    # ============================================================

    HEIGHT = H_PATCHES * P
    WIDTH = W_PATCHES * P

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    print(f"加载图片: {IMAGE_PATH}")
    img_tensor, original_size = load_image(IMAGE_PATH, HEIGHT, WIDTH)
    print(f"原始尺寸: {original_size}, 处理尺寸: {WIDTH}x{HEIGHT}")

    if DEVICE != 'cpu' and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        device = DEVICE
    else:
        device = 'cpu'
    print(f"使用设备: {device}")

    # ---- 1. 掩码图 ----
    print("生成掩码图...")
    x = patchify(img_tensor)
    _, mask, _ = random_masking(x, MASK_RATIO, device=device)
    mask_path = os.path.join(OUTPUT_DIR, f"{basename}_mask.png")
    save_mask_image(mask, H_PATCHES, W_PATCHES, P, mask_path)
    print(f"掩码图已保存: {mask_path}")

    # ---- 2. 重建图 ----
    print("生成重建图...")
    x = patchify(img_tensor)
    x_masked, mask, ids_restore = random_masking(x, MASK_RATIO, device=device)
    N, L, D = x.shape
    x_recon = torch.zeros(N, L, D, device=device)
    len_keep = x_masked.shape[1]
    x_recon[:, :len_keep, :] = x_masked
    x_recon = torch.gather(x_recon, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
    recon_img = unpatchify(x_recon)
    recon_path = os.path.join(OUTPUT_DIR, f"{basename}_recon.png")
    save_tensor_image(recon_img.cpu(), recon_path)
    print(f"重建图已保存: {recon_path}")

    # ---- 3. 增强图 ----
    print("生成增强图...")
    x = patchify(img_tensor)
    uncertainty = x.std(dim=-1)
    uncertainty_mask = deal_uncertainty_patch(uncertainty).to(device)
    x_enhanced = x.clone()
    x_enhanced[uncertainty_mask == 1] = 0
    enhanced_img = unpatchify(x_enhanced)
    enhanced_path = os.path.join(OUTPUT_DIR, f"{basename}_enhanced.png")
    save_tensor_image(enhanced_img.cpu(), enhanced_path)
    print(f"增强图已保存: {enhanced_path}")

    print("全部完成!")


if __name__ == '__main__':
    main()
