# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
from .contmix import ContMixBlock


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1, use_contmix=True):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.use_contmix = use_contmix

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        
        if self.use_contmix:
            # 使用 ContMixBlock 替代第一个卷积层
            self.contmix_block = ContMixBlock(dim=num_input_features * 256, kernel_size=7, smk_size=5, num_heads=2, mlp_ratio=4)
            self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 1, stride)  # 降维用的 1x1 卷积
        else:
            self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
            
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))
        if self.use_contmix:
            self.net.append(self.contmix_block)

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        
        if self.use_contmix:
            # 先通过 ContMixBlock，再降维
            out = self.contmix_block(out)
            out = self.convs[("pose", 0)](out)
            # 处理后续层
            out = self.relu(out)
            out = self.convs[("pose", 1)](out)
            intermediate_feature = out
            out = self.relu(out)
            out = self.convs[("pose", 2)](out)
        else:
            # 原有的流程
            for i in range(3):
                out = self.convs[("pose", i)](out)
                if i == 1:
                    intermediate_feature = out
                if i != 2:
                    out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation, intermediate_feature