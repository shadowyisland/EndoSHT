# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import *
from .FMBConv import FMBPlusPlus
from .LDAM import LDAM


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        self.ldam_modules = nn.ModuleList()
        
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # 为跳跃连接添加 LDAM 模块
            if self.use_skips and i > 0:
                self.ldam_modules.append(
                    LDAM(
                        enc_dim=self.num_ch_enc[i - 1],
                        dec_dim=num_ch_out,
                        out_dim=self.num_ch_enc[i - 1]
                    )
                )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = FMBPlusPlus(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        ldam_idx = 0
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            upsampled_x = upsample(x, mode=self.upsample_mode)
            x = [upsampled_x]
            if self.use_skips and i > 0:
                # 使用 LDAM 模块融合编码器和解码器特征
                fused_feat = self.ldam_modules[ldam_idx](input_features[i - 1], upsampled_x)
                x += [fused_feat]
                ldam_idx += 1
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs