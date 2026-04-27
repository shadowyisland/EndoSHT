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
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from PIL import Image
import cv2

from depthnet.networks.layers import disp_to_depth
from depthnet.utils import readlines
import depthnet.datasets as datasets
import depthnet.networks as networks
from depthnet.datasets import C3VDDataset

# path of model
# model_name = "RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0"
# model_name = "c3vd_v2/ablation/supervised_depthnet极市3vd_v2_monovit_rep2"
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
cfgs.update({'device': device, "not_load_nets": ["net_pose_encoder", "net_pose_decoder", "net_depth_intrinsics"]})

# 创建保存深度图的目录
save_dir = "./visual_depth_maps"
os.makedirs(save_dir, exist_ok=True)


def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
    return disp_color


def getDepthNet():
    # cp_path = Path('results') / model_name / "models" / "weights_18"
    cp_path = Path('newresults5') / model_name / "models" / "weights_23"
    # cp_path = Path极市results') / model_name / "models" / "weights_last"

    model = globals().get(cfgs.get('model'))(cfgs)

    cp = {}
    for network_name in model.network_names:
        cp[network_name] = torch.load(cp_path / "{}.pth".format(network_name), map_location=device)

    model.load_model_state(cp)

    return model


model = getDepthNet()
model.to_device(device)

exps = [cfgs["model_name"]]

modes = ['test_seperately/test_cecum_t4_b_under_review',
         'test_seperately/test_desc_t4_a_under_review',
         'test_seperately/test_sigmoid_t3_b_under_review',
         'test_seperately/test_trans_t4_b_under_review',
         ]

test_mean_errors = []
meds = []
stds = []
masses = []

for mode in modes:

    gt_depths = []
    split = cfgs.get('split')
    fpath = os.path.join("splits", split, "{}_files.txt")
    data_path = cfgs.get('data_path')
    filenames = readlines(fpath.format(mode))
    for line in tqdm(filenames):
        folder, _, frame_id, _ = line.split()
        frame_fpath = os.path.join(data_path, folder, "{}_depth.tiff".format(frame_id))
        gt_depth = cv2.imread(frame_fpath, -1) / (2 ** 16)
        gt_depths.append(gt_depth)

    print(cfgs["model_name"])

    num_workers = cfgs.get('num_workers', 4)
    height = cfgs.get('height', 256)
    width = cfgs.get('width', 320)
    frame_ids = [0]
    num_scales = 1
    dataset = globals().get(cfgs.get('dataset', C3VDDataset))
    split = cfgs.get('split')
    fpath = os.path.join("splits", split, "{}_files.txt")
    filenames = readlines(fpath.format(mode))
    img_ext = '.png' if cfgs.get('png', False) else '.jpg'
    matcher_result_load = None  # np.load(cfgs.get('matcher_result', None), allow_pickle=True).all()

    dataset = dataset(data_path, filenames, matcher_result_load,
                      height, width, frame_ids, num_scales,
                      is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

    model.set_eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(width, height))

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            input_color = data[("color", 0, 0)].to(device)

            output = model.net_depth_decoder(model.net_depth_encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

            # 保存深度图
            for i in range(pred_disp.shape[0]):
                single_pred_disp = pred_disp[i]
                single_pred_disp = cv2.resize(single_pred_disp, (1350, 1080))

                # 获取对应的文件名
                file_idx = idx * dataloader.batch_size + i
                if file_idx < len(dataset.filenames):
                    raw_line = dataset.filenames[file_idx]
                    parts = raw_line.split()
                    if len(parts) >= 4:
                        folder, _, frame_id, _ = parts
                        # 创建文件名
                        vis_file_name = os.path.join(
                            save_dir,
                            f"{folder.replace('/', '_')}_{frame_id}_depth.png"
                        )
                        # 渲染并保存深度图
                        vis_pred_depth = render_depth(single_pred_disp)
                        cv2.imwrite(vis_file_name, vis_pred_depth)

            # input_colors.append(data[("color", 0, 0)].numpy())
    pred_disps = np.concatenate(pred_disps)

    MIN_DEPTH = 0.001
    MAX_DEPTH = 1.
    errors = []
    ratios = []


    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    print("gt_width, gt_height", gt_width, gt_height)

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    test_mean_errors.append(mean_errors)
    meds.append(med)
    stds.append(np.std(ratios / med))
    masses.append(len(dataset))

print(cfgs["model_name"])
print("meds: " + ("{:0.3f}  " * len(meds)).format(*meds))
mean_med = np.sum([mass * med for mass, med in zip(masses, meds)]) / np.sum(masses)
# print("mean med: " + ("{:0.3f}").format(mean_med))
mean_std = np.sum([mass * std for mass, std in zip(masses, stds)]) / np.sum(masses)
# print("mean std: {}".format(mean_std))
total_mean_errors = np.sum([mass * mean_errors for mass, mean_errors in zip(masses, test_mean_errors)],
                           axis=0) / np.sum(masses)

print("\n  " + ("{:>8} | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "med", "std"))
print(
    ("&{: 8.3}  " * 7).format(*total_mean_errors.tolist()) + "&{: 8.3}  &{: 8.3}  ".format(mean_med, mean_std) + "\\\\")
print("\n-> Done!")