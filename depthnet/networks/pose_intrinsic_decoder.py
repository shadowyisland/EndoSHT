# --- 基于 OverLoCK (CVPR 2025) 缝合的位姿与内参预测网络 ---
# 创新点包装：MS-OverCalib (Multi-Scale Overview Calibration Network)
# 核心逻辑：利用大感受野总览机制捕捉全局运动特征，结合上下文混合动态预测相机内参。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 如果你的 layers.py 在同级目录，请确保 transformation_from_parameters 可用
# from .layers import transformation_from_parameters

class DilatedReparamBlock(nn.Module):
    """
    OverLoCK 核心组件的简化版：空洞重参数化块
    包装：通过并行空洞卷积模拟超大感受野（Large Kernel），有效捕捉内窥镜边缘的畸变特征。
    """

    def __init__(self, channels):
        super(DilatedReparamBlock, self).__init__()
        # 主路径：标准 3x3 深度卷积
        self.main_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        # 缝合路径 1：空洞率 2
        self.dil_conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False)

        # 缝合路径 2：空洞率 4
        self.dil_conv4 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4, groups=channels, bias=False)

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # 多路径融合体现“多尺度”
        out = self.main_conv(x) + self.dil_conv2(x) + self.dil_conv4(x)
        return self.act(self.bn(out))


class ContextMixingBlock(nn.Module):
    """
    OverLoCK 创新机制：上下文混合模块
    包装：基于“先全局总览后局部精读”的理念，动态调节空间特征权重。
    """

    def __init__(self, dim):
        super(ContextMixingBlock, self).__init__()
        # Overview 分支：捕捉全局上下文
        self.overview = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        # Focus 分支：精读局部特征
        self.focus = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        # 全局语义指导局部精调
        return x * self.overview(x) + self.focus(x)


class MSOverCalibDecoder(nn.Module):
    """
    完整解码器模块
    缝合点：OverLoCK + 动态回归头
    作用：同时输出 6-DoF 位姿和 4 参数相机内参 (fx, fy, cx, cy)
    """

    def __init__(self, num_ch_enc, num_frames_to_predict_for=2, stride=1):
        super(MSOverCalibDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_frames_to_predict_for = num_frames_to_predict_for

        # 接收编码器最后一层特征 (Bottleneck)
        input_dim = num_ch_enc[-1]

        # --- 缝合 OverLoCK 处理层 ---
        self.pre_process = nn.Sequential(
            nn.Conv2d(input_dim, 256, 1),
            DilatedReparamBlock(256),
            ContextMixingBlock(256)
        )

        # --- 位姿预测头 (Pose Head) ---
        self.pose_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        )

        # --- 内参预测头 (Intrinsic Head) ---
        # 包装话术：自适应几何校准分支，利用全局池化聚合语义信息进行内参回归
        self.intrinsic_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid()  # 限制在 0-1 之间，后续在 model.py 中映射
        )

    def forward(self, input_features):
        # 健壮性检查：如果输入是嵌套列表 [[f1, f2...]]，则取内部列表
        if isinstance(input_features, list) and isinstance(input_features[0], list):
            input_features = input_features[0]

        # 提取编码器的 Bottleneck 特征
        last_features = input_features[-1]

        # 经过 OverLoCK 增强
        enhanced_feat = self.pre_process(last_features)

        # 1. 预测位姿 (Pose)
        out_pose = self.pose_conv(enhanced_feat)
        out_pose = out_pose.mean(3).mean(2)  # 全局平均池化
        out_pose = 0.01 * out_pose.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out_pose[:, :, :, :3]
        translation = out_pose[:, :, :, 3:]

        # 2. 预测内参 (Intrinsics)
        # 输出为比例因子：[fx_ratio, fy_ratio, cx_ratio, cy_ratio]
        raw_intrinsics = self.intrinsic_head(enhanced_feat)

        return axisangle, translation, raw_intrinsics