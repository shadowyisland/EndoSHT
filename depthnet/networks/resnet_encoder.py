# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

# 1. 添加预训练模型URL的硬编码字典
RESNET_URLS = {
    18: "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    50: "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        # 修改输入通道数以支持多图像输入
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        try:
            # 2. 新版本torchvision的加载方式 (≥0.13)
            weights = getattr(models, f"ResNet{num_layers}_Weights").DEFAULT
            state_dict = getattr(models, f"resnet{num_layers}")(weights=weights).state_dict()
        except AttributeError:
            try:
                # 3. 旧版本torchvision的加载方式 (<0.13)
                url = models.resnet.model_urls[f'resnet{num_layers}']
                state_dict = model_zoo.load_url(url)
            except AttributeError:
                # 4. 最终回退方案：使用硬编码URL
                state_dict = model_zoo.load_url(RESNET_URLS[num_layers])

        # 5. 调整第一层卷积权重以适应多图像输入
        conv1_weight = state_dict['conv1.weight']
        # 复制权重通道以适应输入图像数量
        conv1_weight = torch.cat([conv1_weight] * num_input_images, dim=1)
        # 平均权重以保持数值稳定性
        conv1_weight = conv1_weight / num_input_images
        state_dict['conv1.weight'] = conv1_weight

        # 6. 加载调整后的权重
        model.load_state_dict(state_dict)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        # 7. 图像归一化参数 (ImageNet标准)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # 8. 各层输出通道数
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            # 9. 多图像输入使用自定义模型
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            # 10. 单图像输入的预训练模型加载
            if pretrained:
                try:
                    # 新版本加载方式
                    weights = getattr(models, f"ResNet{num_layers}_Weights").DEFAULT
                    self.encoder = resnets[num_layers](weights=weights)
                except AttributeError:
                    # 旧版本加载方式
                    self.encoder = resnets[num_layers](pretrained=True)
            else:
                # 不使用预训练权重
                self.encoder = resnets[num_layers](pretrained=False)

        # 11. 深层ResNet的通道数调整
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        """前向传播提取特征"""
        self.features = []
        # 12. 图像预处理 (近似ImageNet归一化)
        x = (input_image - 0.45) / 0.225

        # 13. 特征提取流程
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features