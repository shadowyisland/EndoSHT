import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCA_Intrinsics(nn.Module):
    """
    创新点模块：多尺度上下文感知内参估计网络 (MSCA-Intrinsics)
    缝合技巧：OverLoCK (CVPR 2025) 的 Overview-Net (全局) + Focus-Net (局部) 逻辑
    包装话术：针对内窥镜视场狭窄且镜头参数动态波动的痛点，本模块模仿视觉系统的“先全局后局部”机制。
             Overview路径捕获管腔整体几何结构，Focus路径提取粘膜纹理的精细特征，两者协同回归相机内参矩阵。
    """

    def __init__(self, encoder_chans, width, height):
        super(MSCA_Intrinsics, self).__init__()
        self.width = width
        self.height = height

        # 1. Overview Path (全局总览): 处理最深层特征 (Stage 4)
        # 捕捉管腔的全局几何轮廓
        self.overview_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # 压缩到 4x4 极小分辨率获取全局语义
            nn.Conv2d(encoder_chans[-1], 256, 1),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True)
        )

        # 2. Focus Path (局部聚焦): 处理中间层特征 (Stage 2)
        # 捕捉近处组织的纹理细节
        self.focus_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # 压缩到 8x8 兼顾空间局部性
            nn.Conv2d(encoder_chans[2], 256, 1),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True)
        )

        # 3. 动态融合与回归
        # 回归 4 个参数: [fx, fy, cx, cy]
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Softplus()  # 确保内参始终为正
        )

    def forward(self, features):
        """
        features: 编码器输出的 5 层特征列表
        """
        # A. 全局总览特征
        global_ctx = self.overview_conv(features[-1])  # 使用 Stage 4

        # B. 局部聚焦特征
        local_ctx = self.focus_conv(features[2])  # 使用 Stage 2

        # C. 特征融合
        combined = torch.cat([global_ctx, local_ctx], dim=1)

        # D. 回归相机内参 (基于图像尺寸进行缩放归一化)
        params = self.regressor(combined)

        # 包装成内参矩阵 K
        # params: [batch, 4] -> [fx, fy, cx, cy]
        K = torch.zeros((params.shape[0], 3, 3), device=params.device)

        # 经验公式：初始化焦距约为图像宽度的 0.5-0.8 倍
        K[:, 0, 0] = params[:, 0] * self.width
        K[:, 1, 1] = params[:, 1] * self.height
        K[:, 0, 2] = params[:, 2] * self.width
        K[:, 1, 2] = params[:, 3] * self.height
        K[:, 2, 2] = 1.0

        return K