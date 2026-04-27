import os

os.chdir("/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main")
from depthnet.utils import *
from depthnet.model import EstimateDepth

from pathlib import Path
import os
import torch
from torch.utils.data.dataloader import default_collate
import collections

import cv2
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch.utils.data import DataLoader

from PIL import Image
import cv2

from depthnet.networks.layers import disp_to_depth
from depthnet.utils import readlines
import depthnet.datasets as datasets
import depthnet.networks as networks
from depthnet.datasets import C3VDDataset
import time
from depthnet.datasets.gastro_dataset import get_data_loaders

# path of model
# model_name = "RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0"
# model_name = "c3vd_v2/ablation/supervised_depthnet_c3vd_v2_monovit_rep2"
# model_name = "RC_baseline_depthnet_c3vd_v2_monodepth2"
model_name = "RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2"
device = torch.device("cuda")
gpu_id = 0
"""
if gpu_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
"""
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

cfgs = load_yaml(
    "/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main/results/RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0/models/configs.yml")
cfgs.update({'device': device, "not_load_nets": ["net_pose_encoder", "net_pose_decoder"]})


def getDepthNet():
    # cp_path = Path('results') / model_name / "models" / "weights_18"
    cp_path = Path('results2') / model_name / "models" / "weights_24"
    # cp_path = Path('results') / model_name / "models" / "weights_last"

    model = globals().get(cfgs.get('model'))(cfgs)

    cp = {}
    for network_name in model.network_names:
        cp[network_name] = torch.load(cp_path / "{}.pth".format(network_name), map_location=device)

    model.load_model_state(cp)

    return model


model = getDepthNet()

model.to_device(device)

inference_times = []

dummy_input = torch.randn(8,3,256,320)


with torch.no_grad():
    for _ in range(100):
        start_time = time.time()
        _ = model.net_depth_decoder(model.net_depth_encoder(dummy_input))
        total_time += time.time() - start_time
avg_time = total_time/100
print(avg_time*1000)
