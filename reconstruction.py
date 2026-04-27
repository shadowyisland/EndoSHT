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
import open3d as o3d
from PIL import Image
import cv2

from depthnet.networks.layers import disp_to_depth
from depthnet.utils import readlines
import depthnet.datasets as datasets
import depthnet.networks as networks
from depthnet.datasets import C3VDDataset

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
cfgs.update({'device': device, "not_load_nets": ["net_pose_encoder", "net_pose_decoder", "net_depth_intrinsics"]})


def getDepthNet():
    # cp_path = Path('results') / model_name / "models" / "weights_18"
    cp_path = Path('newresultsbest_bestcamera_attention_lightweightdepthdecoder') / model_name / "models" / "weights_23"

    # cp_path = Path('newresultsbest_bestcamera_attention') / model_name / "models" / "weights_23"
    # cp_path = Path('results') / model_name / "models" / "weights_last"
    # cp_path = Path('newresults_0.15_0.15') / model_name / "models" / "weights_23"

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
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

    model.set_eval()

    pred_disps = []
    rgbs = []
    cam_Ks = []

    print("-> Computing predictions with size {}x{}".format(width, height))

    with torch.no_grad():
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].to(device)

            output = model.net_depth_decoder(model.net_depth_encoder(input_color))
            rgbs.append(input_color)
            cam_Ks.append(data[("K", 0)])

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)
            # input_colors.append(data[("color", 0, 0)].numpy())
    pred_disps = np.concatenate(pred_disps)

    MIN_DEPTH = 0.001
    MAX_DEPTH = 100.
    errors = []
    ratios = []
    pcds = []


    def compute_scale(gt, pred, min, max):
        mask = np.logical_and(gt > min, gt < max)
        pred = pred[mask]
        gt = gt[mask]
        scale = np.median(gt) / np.median(pred)
        return scale


    def reconstruct_pointcloud(rgb, depth, cam_K, vis_rgbd=False):

        rgb = np.asarray(rgb, order="C")
        rgb_im = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth)

        print(rgb_im)
        print(depth_im)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im,
                                                                        convert_rgb_to_intensity=False)
        if vis_rgbd:
            plt.subplot(1, 2, 1)
            plt.title('RGB image')
            plt.imshow(rgbd_image.color)
            plt.subplot(1, 2, 2)
            plt.title('Depth image')
            plt.imshow(rgbd_image.depth)
            plt.colorbar()
            plt.show()

        cam = o3d.camera.PinholeCameraIntrinsic()
        cam.intrinsic_matrix = cam_K

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            cam
        )

        return pcd


    save_root = os.path.join(os.getcwd(), "reconstructions")  # 根目录，可改为你想要的路径
    save_dir = os.path.join(save_root, cfgs["model_name"], mode.replace("/", "_"))
    os.makedirs(save_dir, exist_ok=True)
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        # pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        pred_height, pred_width = pred_depth.shape[:2]
        gt_depth = cv2.resize(gt_depth, (pred_width, pred_height), interpolation=cv2.INTER_NEAREST)

        scale = compute_scale(gt_depth, pred_depth, MIN_DEPTH, MAX_DEPTH)

        pred_depth *= scale

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        rgb = rgbs[i].squeeze().permute(1, 2, 0).cpu().numpy() * 255
        cam_K = cam_Ks[i][0, :3, :3].numpy()
        pcd = reconstruct_pointcloud(rgb, pred_depth, cam_K, vis_rgbd=False)
        o3d.visualization.draw_geometries([pcd])
        try:
            line = filenames[i]
            folder, _, frame_id, _ = line.split()  # 与你文件格式一致：folder id1 id2 id3 -> frame_id = id2
            seq_name = folder.replace("/", "_").replace("\\", "_")
            frame_id_padded = frame_id.zfill(6)
        except Exception:
            seq_name = "unknown"
            frame_id_padded = str(i).zfill(6)
        # o3d.visualization.draw_geometries([pcd])
        fn = os.path.join(save_dir, f"{seq_name}_{frame_id_padded}.ply")
        o3d.io.write_point_cloud(fn, pcd)
        if i == 0:
            break
    print('Saving point clouds...')

print(cfgs["model_name"])
