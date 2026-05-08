import torch
import torch.nn as nn
from .LAE import LAE
from .modifyppm import ModifyPPM
from .contmix import ContMixBlock

class IntrinsicsHead(nn.Module):
    def __init__(self, num_ch_enc, use_contmix=True):
        super(IntrinsicsHead, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.use_contmix = use_contmix
        
        self.convs_suqeeze = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        
        if self.use_contmix:
            self.contmix_block = ContMixBlock(dim=256, kernel_size=7, smk_size=5, num_heads=2, mlp_ratio=4)
            
        self.focal_length_conv = nn.Conv2d(256, 2, 1, bias=False)
        self.offsets_conv = nn.Conv2d(256, 2, 1, bias=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.lae = LAE(256)
        self.ppm = ModifyPPM(256,64,[3,6,9,12])

    def forward(self, bottleneck, img_width, img_height):
        curr_device = bottleneck.device
        batch_size = bottleneck.shape[0]
        intrinsics_mat = torch.eye(4).unsqueeze(0).to(curr_device)
        intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)
        
        # 只有当输入通道不是256时才进行squeeze转换
        if bottleneck.shape[1] != 256:
            bottleneck = self.convs_suqeeze(bottleneck)
        
        if self.use_contmix:
            bottleneck = self.contmix_block(bottleneck)
            
        bottleneck = self.ppm(bottleneck)
        bottleneck = self.lae(bottleneck)

        bottleneck = self.global_pooling(bottleneck)
        focal_lengths = (
                ((self.softplus(self.focal_length_conv(bottleneck).squeeze())) + 0.5) *
                torch.Tensor([img_width, img_height]).to(curr_device)
        )
        offsets = (
                (self.offsets_conv(bottleneck).squeeze() + 0.5) *
                torch.Tensor([img_width, img_height]).to(curr_device)
        ).unsqueeze(-1)
        foci = torch.diag_embed(focal_lengths)

        intrinsics_mat[:, :2, :2] = foci
        intrinsics_mat[:, :2, 2:3] = offsets

        return intrinsics_mat