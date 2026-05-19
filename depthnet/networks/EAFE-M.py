from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class EAFEM:
    """
    EAFE-M: 自监督单目深度估计的增强特征增强模块

    该模块实现论文中描述的四个核心机制：
    1. 双重掩模策略 (Dual Mask Strategy): 镜面掩模 + 不确定性掩模
    2. 上下文感知重建 (Context-aware Reconstruction): ReconNet重建网络
    3. 特征增强 (Feature Enhancement): 掩码引导的融合策略
    4. 前向与损失分离机制 (Forward-Loss Separation)

    论文公式:
        - 镜面掩模: m_s = {p | brightness(p) > 240 且 消色差}
        - 统一掩模: M = m_s ∪ m_u
        - 掩码图像: I_M = I_s ⊙ (1-M)
        - 融合图像: I_A = W·I_R + (1-W)·I_s
        - 损失函数: L_repro = (1/N)Σ(1-M_t)·[α/2(1-SSIM) + (1-α)|I_t - Î_t|]
    """

    def __init__(self, height=256, width=320, device='cuda'):
        """
        初始化EAFE-M模块

        Args:
            height: 输入图像高度
            width: 输入图像宽度
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.height = height
        self.width = width
        self.device = device

        # ReconNet重建网络：三层深度可分离卷积，无跳跃连接
        self.recon_net = ReconNet().to(device)

        # SSIM与L1损失的平衡系数（论文默认值）
        self.alpha = 0.85

    def generate_specular_mask(self, image, brightness_threshold=240):
        """
        生成镜面掩模 m_s

        论文描述: 针对内镜强光源产生的干涉，通过物理特征提取生成镜面掩模，
        将亮度超过特定阈值（如240）且呈现消色差特性的像素判定为反射伪影。

        Args:
            image: 输入图像 (N, 3, H, W)，值范围 [0, 1]
            brightness_threshold: 亮度阈值，默认240（针对[0,255]范围）

        Returns:
            specular_mask: 镜面掩模 (N, 1, H, W)，值为0（有效）或1（无效/镜面反射）
        """
        # 将[0,1]范围转换为[0,255]范围进行阈值判断
        image_255 = image * 255.0

        # 计算亮度: L = 0.299*R + 0.587*G + 0.114*B（标准亮度公式）
        brightness = 0.299 * image_255[:, 0:1, :, :] + \
                     0.587 * image_255[:, 1:2, :, :] + \
                     0.114 * image_255[:, 2:3, :, :]

        # 条件1: 亮度超过阈值（高光区域）
        bright_pixels = brightness > brightness_threshold

        # 条件2: 消色差特性 - 即RGB三通道值接近相等
        # 计算RGB通道间的差异
        r, g, b = image_255[:, 0:1, :, :], image_255[:, 1:2, :, :], image_255[:, 2:3, :, :]
        rg_diff = torch.abs(r - g)
        rb_diff = torch.abs(r - b)
        gb_diff = torch.abs(g - b)
        max_diff = torch.max(torch.max(rg_diff, rb_diff), gb_diff)

        # 消色差阈值：通道间差异小于一定值视为消色差
        achromatic_threshold = 15.0
        achromatic_pixels = max_diff < achromatic_threshold

        # 镜面反射区域：同时满足高亮和消色差条件
        specular_mask = (bright_pixels & achromatic_pixels).float()

        return specular_mask

    def generate_uncertainty_mask(self, image, encoder, num_samples=8):
        """
        生成不确定性掩模 m_u

        论文描述: 为了捕捉难以通过静态阈值定义的无效区域（如阴影、粘液区或运动模糊），
        通过对原始图像施加多次随机掩模并送入网络，计算像素级预测的标准差，
        标准差较高的区域反映了模型对该处空间结构感知的脆弱性。

        Args:
            image: 输入图像 (N, 3, H, W)
            encoder: 预训练的特征编码器，用于提取特征
            num_samples: 随机掩模采样次数，默认8次

        Returns:
            uncertainty_mask: 不确定性掩模 (N, 1, H, W)，值为0（有效）或1（无效/高不确定性）
        """
        N, C, H, W = image.shape
        patch_size = 32

        # 验证图像尺寸是否被patch_size整除
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"图像尺寸({H}x{W})必须被patch_size({patch_size})整除"

        h_patches = H // patch_size
        w_patches = W // patch_size
        num_patches = h_patches * w_patches

        # 存储每次采样的深度预测
        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                # 生成随机掩码比例 (0.5 ~ 0.85)
                mask_ratio = random.uniform(0.5, 0.85)

                # 创建随机噪声用于打乱patch顺序
                noise = torch.rand(N, num_patches, device=self.device)
                ids_shuffle = torch.argsort(noise, dim=1)
                ids_restore = torch.argsort(ids_shuffle, dim=1)

                # 计算保留的patch数量
                len_keep = int(num_patches * (1 - mask_ratio))

                # 生成二进制掩码：0保留，1移除
                mask = torch.ones(N, num_patches, device=self.device)
                mask[:, :len_keep] = 0
                mask = torch.gather(mask, dim=1, index=ids_restore)

                # 将patch级别的掩码转换为图像级别的掩码
                mask_image = mask.view(N, 1, h_patches, w_patches)
                mask_image = mask_image.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)

                # 应用掩码到图像
                masked_image = image * (1 - mask_image)

                # 通过编码器提取特征
                features = encoder(masked_image)

                # 取最后一层特征计算不确定性（这里用特征方差近似）
                # 实际应用中可以通过一个轻量解码器获取深度预测
                if isinstance(features, list):
                    feature = features[-1]
                else:
                    feature = features

                # 对特征图计算像素级标准差
                feature_std = torch.std(feature, dim=1, keepdim=True)

                # 上采样到原始图像尺寸
                feature_std = F.interpolate(feature_std, size=(H, W), mode='bilinear', align_corners=False)

                predictions.append(feature_std)

        # 计算像素级标准差的均值（跨多次采样）
        predictions = torch.stack(predictions, dim=0)  # (num_samples, N, 1, H, W)
        uncertainty_map = torch.std(predictions, dim=0)  # (N, 1, H, W)

        # 归一化不确定性图到[0,1]
        uncertainty_map_min = uncertainty_map.view(N, -1).min(dim=1)[0].view(N, 1, 1, 1)
        uncertainty_map_max = uncertainty_map.view(N, -1).max(dim=1)[0].view(N, 1, 1, 1)
        uncertainty_map = (uncertainty_map - uncertainty_map_min) / (uncertainty_map_max - uncertainty_map_min + 1e-8)

        # 设置不确定性阈值：标准差前25%的区域判定为高不确定性（无效区域）
        threshold = 0.75
        uncertainty_mask = (uncertainty_map > threshold).float()

        return uncertainty_mask

    def generate_combined_mask(self, image, encoder=None):
        """
        生成统一掩模 M = m_s ∪ m_u

        论文描述: 通过逻辑或操作合并镜面掩模和不确定性掩模，形成统一掩模，
        该掩模在损失计算中起到屏蔽噪声信号的关键作用。

        Args:
            image: 输入图像 (N, 3, H, W)
            encoder: 可选，用于不确定性掩模计算的编码器

        Returns:
            combined_mask: 统一掩模 (N, 1, H, W)，值为0（有效区域）或1（无效区域）
        """
        # 生成镜面掩模
        specular_mask = self.generate_specular_mask(image)

        # 如果提供了编码器，则同时生成不确定性掩模
        if encoder is not None:
            uncertainty_mask = self.generate_uncertainty_mask(image, encoder)
            # 逻辑或操作合并
            combined_mask = torch.clamp(specular_mask + uncertainty_mask, min=0, max=1)
        else:
            combined_mask = specular_mask

        return combined_mask

    def forward(self, image, encoder=None, return_mask=False):
        """
        EAFE-M完整前向传播

        论文描述:
        1. 前向路径中，网络利用增强图像I_A进行预测
        2. 通过EAFE-M生成的掩码屏蔽无效区域
        3. 重建网络对被掩模屏蔽的无效区域进行几何恢复
        4. 掩码引导的融合策略生成增强图像

        Args:
            image: 输入原始图像 (N, 3, H, W)
            encoder: 可选，用于不确定性掩模计算的编码器
            return_mask: 是否返回中间掩模

        Returns:
            enhanced_image: 增强图像 I_A = W·I_R + (1-W)·I_s
            或者 (enhanced_image, masks_dict) 如果 return_mask=True
        """
        # Step 1: 生成统一掩模 M
        combined_mask = self.generate_combined_mask(image, encoder)

        # Step 2: 创建掩码处理后的图像 I_M = I_s ⊙ (1-M)
        masked_image = image * (1 - combined_mask)

        # Step 3: 通过ReconNet进行上下文感知重建
        reconstructed_output = self.recon_net(masked_image)
        reconstructed_image = reconstructed_output[:, :3, :, :]  # RGB通道
        weight_map = reconstructed_output[:, 3:4, :, :]  # 融合权重图W

        # 权重图通过Sigmoid激活到[0,1]
        weight_map = torch.sigmoid(weight_map)

        # Step 4: 掩码引导的图像融合 I_A = W·I_R + (1-W)·I_s
        enhanced_image = weight_map * reconstructed_image + (1 - weight_map) * image

        if return_mask:
            masks_dict = {
                'combined_mask': combined_mask,
                'masked_image': masked_image,
                'reconstructed_image': reconstructed_image,
                'weight_map': weight_map
            }
            return enhanced_image, masks_dict

        return enhanced_image

    def compute_photometric_loss(self, target_image, reprojected_image, mask):
        """
        计算基于掩模的屏蔽重投影损失

        论文公式:
        L_repro = (1/N)Σ(1-M_t(p))·[α/2(1-SSIM(I_t(p),Î_t(p))) + (1-α)|I_t(p) - Î_t(p)|]

        其中:
        - M_t为目标帧无效掩码
        - α为SSIM与L1损失的平衡系数
        - I_t为目标帧原始图像
        - Î_t为重投影图像
        - N为有效像素总数

        论文强调:
        - 预测信号来源于干净的增强输入
        - 监督信号来源于真实的物理成像（带掩模的原始图像）
        - 这种双路径逻辑有效避免了模型拟合重构过程中可能产生的虚假纹理

        Args:
            target_image: 目标帧原始图像 (N, 3, H, W)
            reprojected_image: 重投影图像 (N, 3, H, W)
            mask: 统一掩模 M (N, 1, H, W)，值为1表示无效区域

        Returns:
            loss: 屏蔽重投影损失（标量）
        """
        # 有效区域掩码：1-M_t(p)，掩模为0的区域才是有效区域
        valid_mask = (1 - mask)

        # 确保mask维度正确
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)

        # 1. SSIM损失
        ssim_loss = self._ssim(target_image, reprojected_image)
        ssim_component = self.alpha / 2 * (1 - ssim_loss)

        # 2. L1损失
        l1_component = (1 - self.alpha) * torch.abs(target_image - reprojected_image)

        # 3. 组合损失
        combined_loss = ssim_component + l1_component

        # 4. 应用有效区域掩码并取平均
        masked_loss = combined_loss * valid_mask
        loss = masked_loss.sum() / (valid_mask.sum() + 1e-8)

        return loss

    def _ssim(self, img1, img2):
        """
        计算结构相似性指标 (SSIM)

        SSIM公式:
        SSIM(x, y) = [(2μ_xμ_y + C1)(2σ_xy + C2)] / [(μ_x^2 + μ_y^2 + C1)(σ_x^2 + σ_y^2 + C2)]

        Args:
            img1: 图像1 (N, C, H, W)
            img2: 图像2 (N, C, H, W)

        Returns:
            ssim_map: SSIM图 (N, 1, H, W)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 计算均值（使用高斯权重）
        mu1 = F.avg_pool2d(img1, kernel_size=3, stride=1, padding=1)
        mu2 = F.avg_pool2d(img2, kernel_size=3, stride=1, padding=1)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # 计算方差和协方差
        sigma1_sq = F.avg_pool2d(img1 ** 2, kernel_size=3, stride=1, padding=1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, kernel_size=3, stride=1, padding=1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=3, stride=1, padding=1) - mu1_mu2

        # SSIM公式
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map


class ReconNet(nn.Module):
    """
    ReconNet: 轻量化的上下文感知重建网络

    论文描述:
    - 采用仅由三层深度可分离卷积组成的对称式架构
    - 为了强制模型利用周围健康组织的上下文信息进行推理并隔绝伪影特征的渗透，
      该子网络移除了所有跳跃连接，仅通过瓶颈层的全局特征表示生成视觉连续的重建图

    输入: 掩模处理后的原始图像 I_M = I_s ⊙ (1-M)
    输出: 重建图像 I_R + 融合权重图W（共4通道）
    """

    def __init__(self):
        super(ReconNet, self).__init__()

        # 深度可分离卷积块定义
        # 使用深度可分离卷积减少参数量，满足实时性要求

        # Encoder: 三层深度可分离卷积，逐步下采样提取全局特征
        self.encoder = nn.Sequential(
            # Layer 1: 输入3通道，输出32通道，下采样2倍
            self._depthwise_separable_conv(3, 32, stride=2),
            # Layer 2: 输入32通道，输出64通道，下采样2倍
            self._depthwise_separable_conv(32, 64, stride=2),
            # Layer 3: 输入64通道，输出128通道，下采样2倍
            self._depthwise_separable_conv(64, 128, stride=2),
        )

        # 瓶颈层: 全局特征整合
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 10)),  # 全局平均池化到固定尺寸
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder: 三层深度可分离卷积，逐步上采样恢复空间分辨率
        self.decoder = nn.Sequential(
            # Layer 1: 上采样2倍
            self._depthwise_separable_conv_up(128, 64),
            # Layer 2: 上采样2倍
            self._depthwise_separable_conv_up(64, 32),
            # Layer 3: 上采样2倍，恢复到原始分辨率
            self._depthwise_separable_conv_up(32, 16),
        )

        # 输出层: 生成RGB图像和融合权重图（共4通道）
        # RGB通道提供重建图像，额外通道生成融合权重图W
        self.output_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),  # 4通道: 3通道RGB + 1通道权重图W
        )

    def _depthwise_separable_conv(self, in_channels, out_channels, stride=1):
        """
        深度可分离卷积块

        深度可分离卷积将标准卷积分解为：
        1. 深度卷积（逐通道）：每个通道单独滤波
        2. 逐点卷积（1x1）：组合通道信息

        这种分解大幅减少参数量和计算量，满足嵌入式平台的实时性要求。

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长（2时为下采样）

        Returns:
            深度可分离卷积模块
        """
        return nn.Sequential(
            # 深度卷积：groups=in_channels，每个输入通道独立卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 逐点卷积：1x1卷积整合通道信息
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _depthwise_separable_conv_up(self, in_channels, out_channels):
        """
        深度可分离卷积上采样块

        使用转置卷积进行上采样，配合深度可分离卷积结构。

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数

        Returns:
            上采样模块
        """
        return nn.Sequential(
            # 转置卷积上采样2倍
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                              groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 逐点卷积调整通道数
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        ReconNet前向传播

        Args:
            x: 掩码处理后的图像 I_M = I_s ⊙ (1-M)，(N, 3, H, W)

        Returns:
            output: 包含重建图像和融合权重图的4通道张量 (N, 4, H, W)
            - output[:, :3, :, :]: 重建图像 I_R
            - output[:, 3:4, :, :]: 融合权重图 W
        """
        # Encoder: 提取全局上下文特征
        enc_features = self.encoder(x)

        # 瓶颈层: 全局特征整合（无跳跃连接，强制学习全局关联性）
        bottleneck_features = self.bottleneck(enc_features)

        # Decoder: 恢复空间分辨率
        dec_features = self.decoder(bottleneck_features)

        # 上采样到原始图像尺寸（如果瓶颈层改变了尺寸）
        dec_features = F.interpolate(dec_features, size=(x.shape[2], x.shape[3]),
                                    mode='bilinear', align_corners=False)

        # 输出层: 生成RGB + 权重图
        output = self.output_conv(dec_features)

        return output


class Trim:
    """
    Trim类: 提供图像patch级别的处理工具函数

    这些工具函数用于MAE风格的随机掩码生成，
    主要服务于不确定性掩模的计算过程。
    """

    @staticmethod
    def patchify(imgs, patch_size=32):
        """
        将图像转换为patch序列

        Args:
            imgs: (N, 3, H, W) 输入图像
            patch_size: patch大小，默认32

        Returns:
            x: (N, L, patch_size**2 *3) patch序列，L = (H/p)*(W/p)
        """
        p = patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    @staticmethod
    def patchify_uncertainty(imgs, patch_size=32):
        """
        将图像转换为patch序列（单通道，用于不确定性计算）

        Args:
            imgs: (N, 1, H, W) 单通道图像
            patch_size: patch大小，默认32

        Returns:
            x: (N, L, patch_size**2) patch序列
        """
        p = patch_size
        assert imgs.shape[1] % p == 0 and imgs.shape[2] % p == 0

        h = imgs.shape[1] // p
        w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p, w, p))
        x = torch.einsum('nhpwq->nhwpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        return x

    @staticmethod
    def unpatchify(x, patch_size=32, h=None, w=None, h_patches=None, w_patches=None):
        """
        将patch序列还原为图像

        Args:
            x: (N, L, patch_size**2 *3) patch序列
            patch_size: patch大小，默认32
            h: 高度方向patch数（兼容旧版调用）
            w: 宽度方向patch数（兼容旧版调用）
            h_patches: 高度方向patch数，如果为None则从L自动推断
            w_patches: 宽度方向patch数，如果为None则从L自动推断

        Returns:
            imgs: (N, 3, H, W) 还原图像
        """
        p = patch_size
        num_patches = x.shape[1]

        # 优先使用 h 和 w 参数（兼容旧版调用）
        if h is not None:
            h_patches = h
        if w is not None:
            w_patches = w

        if h_patches is None or w_patches is None:
            # 从patch总数推断网格尺寸
            h_patches = w_patches = int(num_patches ** 0.5)

        assert h_patches * w_patches == num_patches

        x = x.reshape(shape=(x.shape[0], h_patches, w_patches, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h_patches * p, w_patches * p))
        return imgs

    @staticmethod
    def random_masking(x, mask_ratio, device='cuda'):
        """
        执行per-sample的随机掩码（通过per-sample shuffle实现）

        论文描述: 通过对原始图像施加多次随机掩模并送入网络，
        计算像素级预测的标准差。

        Args:
            x: (N, L, D) 序列，N=batch，L=长度，D=维度
            mask_ratio: 掩码比例
            device: 计算设备

        Returns:
            x_masked: 掩码后的序列 (N, len_keep, D)
            mask: 二进制掩码 (N, L)，0=保留，1=移除
            ids_restore: 恢复索引，用于还原掩码顺序
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # 为每个sample生成随机噪声
        noise = torch.rand(N, L, device=device)

        # 对噪声排序（升序），小值保留，大值移除
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 保留前len_keep个patch
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成二进制掩码：0=保留，1=移除
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


class Aug:
    """
    Aug类: 提供图像增强和融合相关的工具函数

    这些函数服务于EAFE-M的特征增强阶段，
    实现掩码引导的图像融合策略。
    """

    def __init__(self):
        pass

    @staticmethod
    def fuse_recon_and_detail(original_image, reconstructed_image, weight_map):
        """
        融合原始图像和重建图像

        论文公式: I_A = W·I_R + (1-W)·I_s

        其中:
        - I_A为最终的增强图像
        - I_R为重建图像（提供全局几何骨架）
        - I_s为原始图像（补充真实解剖细节）
        - W为融合权重图（[0,1]范围）

        Args:
            original_image: 原始图像 I_s (N, 3, H, W)
            reconstructed_image: 重建图像 I_R (N, 3, H, W)
            weight_map: 融合权重图 W (N, 1, H, W)，已过Sigmoid

        Returns:
            fused_image: 增强图像 I_A (N, 3, H, W)
        """
        # 确保权重图维度正确
        if weight_map.dim() == 3:
            weight_map = weight_map.unsqueeze(1)

        # I_A = W·I_R + (1-W)·I_s
        fused_image = weight_map * reconstructed_image + (1 - weight_map) * original_image

        return fused_image


def disp_to_depth_for_eafe(disp, min_depth=0.1, max_depth=100.0):
    """
    将视差图转换为深度图

    公式: depth = 1 / (min_disp + (max_disp - min_disp) * disp)
    其中: min_disp = 1/max_depth, max_disp = 1/min_depth

    Args:
        disp: 视差图 (N, 1, H, W)
        min_depth: 最小深度
        max_depth: 最大深度

    Returns:
        scaled_disp: 缩放后的视差
        depth: 深度图
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

    