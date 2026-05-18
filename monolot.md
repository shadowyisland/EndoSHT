# 索引: MonoLoT

## 文件: `documentizer_py.py`

```py
import os
import argparse

def collect_code_to_md(root_dir, output_file):
    """
    遍历指定目录下的所有文件，并将 .go 和 .yaml 文件的内容
    按目录结构写入一个 Markdown 文件。

    :param root_dir: 要遍历的根目录。
    :param output_file: 输出的 Markdown 文件名。
    """
    # 获取绝对路径，以便 relpath 正常工作
    root_dir = os.path.abspath(root_dir)

    try:
        with open(output_file, 'w', encoding='utf-8') as md_file:
            # 1. 将根目录的名称作为一级标题
            md_file.write(f"# 索引: {os.path.basename(root_dir)}\n\n")

            # 2. 遍历整个目录树
            for dirpath, _, filenames in os.walk(root_dir):
                # 筛选出当前目录下的 .go 和 .yaml 文件
                # target_files = sorted([f for f in filenames if f.endswith('.py') or f.endswith('.yaml')])
                target_files = sorted([f for f in filenames if f.endswith('.py')])
                # 如果当前目录中没有目标文件，则跳过
                if not target_files:
                    continue

                # 3. 根据目录深度生成 Markdown 标题
                relative_dir_path = os.path.relpath(dirpath, root_dir)

                # 如果不是根目录本身，才创建目录标题
                if relative_dir_path != ".":
                    # 计算深度，根目录的下一级为 H2
                    depth = relative_dir_path.count(os.sep) + 2
                    heading = '#' * depth
                    md_file.write(f"{heading} 目录: `{relative_dir_path}`\n\n")

                # 4. 遍历并写入每个文件的内容
                for filename in target_files:
                    file_path = os.path.join(dirpath, filename)

                    # 确定文件标题的级别，比其所在目录的标题深一级
                    if relative_dir_path == ".":
                        file_heading_level = 2  # 根目录下的文件使用 H2
                    else:
                        file_heading_level = relative_dir_path.count(os.sep) + 3

                    file_heading = '#' * file_heading_level
                    md_file.write(f"{file_heading} 文件: `{filename}`\n\n")

                    # 根据文件扩展名确定代码块的语言
                    lang = ''
                    if filename.endswith('.py'):
                        lang = 'py'
                    elif filename.endswith('.yaml'):
                        lang = 'yaml'

                    md_file.write(f"```{lang}\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as code_file:
                            md_file.write(code_file.read())
                    except Exception as e:
                        md_file.write(f"Error reading file: {e}")
                    md_file.write("\n```\n\n")

        print(f"成功将代码按目录结构写入到 '{output_file}' 文件中。")

    except IOError as e:
        print(f"写入文件时出错: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将一个目录下的所有 .py 和 .yaml 文件的代码按目录结构收集到一个 Markdown 文件中。",
        epilog="例如: python collect_code_hierarchical.py my_project_code.md -d ./my_project"
    )

    parser.add_argument(
        "output_filename",
        help="要创建的 Markdown 文件的名称 (例如: 'output.md')"
    )

    parser.add_argument(
        "-d", "--directory",
        default=".",
        help="要遍历的根目录 (默认为当前目录)"
    )

    args = parser.parse_args()
    collect_code_to_md(args.directory, args.output_filename)
```

## 文件: `eval1.py`

```py
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

# path of model

#model_name = "RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2"
# model_name = "c3vd_v2/ablation/supervised_depthnet_c3vd_v2_monovit_rep2"
# model_name = "RC_baseline_depthnet_c3vd_v2_monodepth2"
model_name = "RC_matching_depthnet_c3vd_v2_litemono_sd0"

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


cfgs = load_yaml("/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main/results/RC_matching_depthnet_c3vd_v2_litemono_sd0/models/configs.yml")
cfgs.update({'device': device, "not_load_nets": ["net_pose_encoder", "net_pose_decoder"]})



def getDepthNet():
    # cp_path = Path('results') / model_name / "models" / "weights_18"
    cp_path = Path('results') / model_name / "models" / "weights_last"
    # cp_path = Path('results') / model_name / "models" / "weights_last"

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
        gt_depth = cv2.imread(frame_fpath, -1)/(2**16)
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
    matcher_result_load = None #np.load(cfgs.get('matcher_result', None), allow_pickle=True).all()

    dataset = dataset(data_path, filenames, matcher_result_load,
        height, width, frame_ids, num_scales,
        is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

    model.set_eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(width, height))

    with torch.no_grad():
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].to(device)

            output = model.net_depth_decoder(model.net_depth_encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)
            # input_colors.append(data[("color", 0, 0)].numpy())
    pred_disps = np.concatenate(pred_disps)

    MIN_DEPTH = 0.001
    MAX_DEPTH = 1.
    errors = []
    ratios = []

    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
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
total_mean_errors = np.sum([mass * mean_errors for mass, mean_errors in zip(masses, test_mean_errors)], axis=0) / np.sum(masses)

print("\n  " + ("{:>8} | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "med", "std"))
print(("&{: 8.3}  " * 7).format(*total_mean_errors.tolist()) + "&{: 8.3}  &{: 8.3}  ".format(mean_med, mean_std) + "\\\\")
print("\n-> Done!")
```

## 文件: `eval1022.py`

```py
# Converted from depth_color.ipynb
# Conversion time: 2025-10-22T07:31:44.132245
# This file was auto-generated by ChatGPT.
# Code cells are preserved. Markdown cells are added as commented blocks.

# --- Code cell 0 ---
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import os

os.chdir("/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main")

# imgs = ["data/c3vd_v2/imgs/cecum_t4_b_under_review/0000_color.png",
#         "data/c3vd_v2/imgs/desc_t4_a_under_review/0100_color.png",
#         "data/c3vd_v2/imgs/sigmoid_t3_b_under_review/0500_color.png",
#         "data/c3vd_v2/imgs/trans_t4_b_under_review/0400_color.png"]

imgs = ["data/c3vd_v2/imgs/cecum_t4_b_under_review/0028_color.png",
        "data/c3vd_v2/imgs/desc_t4_a_under_review/0020_color.png",
        "data/c3vd_v2/imgs/sigmoid_t3_b_under_review/0025_color.png",
        "data/c3vd_v2/imgs/trans_t4_b_under_review/0090_color.png"]

"""
imgs = [
    "data/c3vd_v2/imgs/cecum_t4_b_under_review/0060_color.png",
    # "data/c3vd_v2/imgs/cecum_t4_b_under_review/0000_color.png",
    "data/c3vd_v2/imgs/desc_t4_a_under_review/0030_color.png",
    # "data/c3vd_v2/imgs/desc_t4_a_under_review/0095_color.png",
    "data/c3vd_v2/imgs/sigmoid_t3_b_under_review/0120_color.png",
    # "data/c3vd_v2/imgs/sigmoid_t3_b_under_review/0000_color.png",
    "data/c3vd_v2/imgs/trans_t4_b_under_review/0455_color.png",
    # "data/c3vd_v2/imgs/trans_t4_b_under_review/0480_color.png",
]
"""
for img in imgs:
    fname = os.path.join(
        "/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main", img)
    gt = pil.open(fname.replace("color.png", "depth.tiff"))
    gt = np.array(gt, np.float32) / (2 ** 16 - 1)

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    im = axes.imshow(gt, cmap="jet_r", vmin=0, vmax=1)
    # axes.imshow(1-gt, vmin=0, vmax=1)
    axes.axis('off')
    fname_colormap = "_".join(img.split("/")[-2:] )
    #plt.savefig(os.path.join("gt_color_map", fname_colormap), bbox_inches='tight', pad_inches=0)
# axes[1].imshow(depths[idx].max() - depths[idx], cmap="jet")
# axes[1].axis('off')


# --- Code cell 1 ---
fig, axes = plt.subplots(1, 1)
fig.set_size_inches(10, 6)
im = axes.imshow(gt, cmap="jet_r", vmin=0, vmax=1)
# axes.imshow(1-gt, vmin=0, vmax=1)
axes.axis('off')
plt.colorbar(im, ax=axes, aspect=12, drawedges=False, ticks=[0, 1])

# --- Code cell 2 ---
import os

os.chdir("/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main")
from depthnet.utils import *
from depthnet.model import EstimateDepth
from pathlib import Path
import glob
import torchvision
from torchvision import transforms

model_list = ["RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2"]

#model_list = ["RCC"]
model_pred_depths = []


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


for model_name in model_list:
    # path of model
    # model_name = "RC_matching_depthnet_c3vd_v2_litemono_sd0"

    device = f'cuda:0'
    gpu_id = 0
    if gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    cfgs = load_yaml(
        "/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main/results/RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0/models/configs.yml")
    cfgs.update({'device': device, "not_load_nets": ["net_pose_encoder", "net_pose_decoder", "net_depth_intrinsics"]})


    def getDepthNet():
    # cp_path = Path('results') / model_name / "models" / "weights_18"
        #cp_path = Path('newresultsbest_bestcamera_attention_lightweightdepthdecoder') / model_name / "models" / "weights_23"
        cp_path = Path('newresultsbest_bestcamera_attention') / model_name / "models" / "weights_23"
        #cp_path = Path('newresults_0.15_0.15') / model_name / "models" / "weights_22"
    # cp_path = Path极市results') / model_name / "models" / "weights_last"

        model = globals().get(cfgs.get('model'))(cfgs)

        cp = {}
        for network_name in model.network_names:
            cp[network_name] = torch.load(cp_path / "{}.pth".format(network_name), map_location=device)

        model.load_model_state(cp)

        return model

    model = getDepthNet()
    model.to_device(device)

    # load
    import os
    import torch
    from torchvision import transforms
    import cv2
    import numpy as np
    from tqdm.auto import tqdm
    from PIL import Image

    from depthnet.networks.layers import disp_to_depth
    from depthnet.utils import readlines
    import depthnet.networks as networks
    from depthnet.networks.layers import transformation_from_parameters

    exps = [cfgs["model_name"]]

    height = cfgs.get('height', 256)
    width = cfgs.get('width', 320)
    model.set_eval()

    pred_depths = []

    print("-> Computing predictions with size {}x{}".format(width, height))

    MIN_DEPTH = 0.001
    MAX_DEPTH = 1.

    with torch.no_grad():
        for fname in tqdm(imgs):
            gt_depth = cv2.imread(fname.replace("color.png", "depth.tiff"), -1)
            gt_depth = gt_depth / (2 ** 16 - 1)
            gt_height, gt_width = gt_depth.shape[:2]

            input_color = pil_loader(fname)
            input_color = transforms.Resize((height, width), torchvision.transforms.InterpolationMode.LANCZOS)(
                input_color)

            # input_color2 = cv2.imread(fname)
            # input_color2 = cv2.resize(input_color2, (width, height))
            # rgb = rgb[:, :, ::-1]

            input_color = transforms.ToTensor()(input_color).unsqueeze(0).to(device)
            # input_color2 = transforms.ToTensor()(input_color2).unsqueeze(0).to(device)

            # input_color = cv2.imread(fname)
            # input_color = cv2.resize(input_color, (width, height))
            # # rgb = rgb[:, :, ::-1]
            # input_color = transforms.ToTensor()(input_color).unsqueeze(0).to(device)

            output = model.net_depth_decoder(model.net_depth_encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])

            pred_disp = pred_disp.cpu()[0, 0, :, :].numpy()
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            pred_depths.append(pred_depth)

    model_pred_depths.append(pred_depths)


# --- Code cell 3 ---

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

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]


fig, axes = plt.subplots(len(imgs), 8)
fig.set_size_inches(42, 20)
axes = axes.flatten()

rmse = np.zeros((len(imgs), 6))

for i, model_name in enumerate(model_list):
    print(model_name)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    for j, img in enumerate(imgs):
        pred_depth = model_pred_depths[i][j]
        axes[8 * j + 2 + i].imshow(pred_depth, cmap="jet_r", vmin=0, vmax=1)
        axes[8 * j + 2 + i].axis('off')

        gt_depth = cv2.imread(img.replace("color.png", "depth.tiff"), -1)
        gt_depth = gt_depth / (2 ** 16 - 1)
        # axes.imshow(1-pred_depth, vmin=0, vmax=1)

        mean_errors = compute_errors(gt_depth, pred_depth)
        print(("&{: 8.3}  " * 7).format(*mean_errors))

        # axes[18 * j + i].set_title("{:.3f}".format(mean_errors[2]), y=-0.14, size=36)
        rmse[j, i] = mean_errors[2]
        # fontweight='bold'

        if i == 0:
            # input_color = cv2.imread(fname)
            # rgb = cv2.resize(input_color, (width, height))
            input_color = pil_loader(img)
            axes[8 * j + 0].imshow(input_color)
            axes[8 * j + 0].axis('off')

            # axes[8 * j + 1].imshow(gt_depth, cmap="plasma_r")
            axes[8 * j + 1].imshow(gt_depth, cmap="jet_r", vmin=0, vmax=1)
            axes[8 * j + 1].axis('off')

for j in range(len(imgs)):
    idx = np.argmin(rmse[j])
    for i in range(len(model_list)):
        if i == idx:
            axes[8 * j + 2 + i].set_title("{:.3f}".format(rmse[j, i]), y=-0.16, size=36, fontweight='bold')
        else:
            axes[8 * j + 2 + i].set_title("{:.3f}".format(rmse[j, i]), y=-0.16, size=36)

plt.tight_layout()
plt.show()


# --- Code cell 4 ---
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

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]


for i, model_name in enumerate(model_list):
    print(model_name)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    for j, img in enumerate(imgs):
        pred_depth = model_pred_depths[i][j]
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)
        # axes.imshow(1-pred_depth, vmin=0, vmax=1)
        axes.imshow(pred_depth, cmap="jet_r", vmin=0, vmax=1)
        axes.axis('off')
        fname_colormap = "_".join(model_name.split("/")) + "_".join(img.split("/")[-2:] + ["colormap.png"])
        plt.savefig(os.path.join("pred_color_map", fname_colormap), bbox_inches='tight', pad_inches=0)

        gt_depth = cv2.imread(img.replace("color.png", "depth.tiff"), -1) / (2 ** 16 - 1)
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)
        # axes.imshow(1-pred_depth, vmin=0, vmax=1)
        axes.imshow(pred_depth - gt_depth, cmap="seismic_r", vmin=-1, vmax=1)
        axes.axis('off')
        fname_divergingmap = "_".join(model_name.split("/")) + "_".join(img.split("/")[-2:] + ["divergingmap.png"])
        plt.savefig(os.path.join("pred_diverging_map", fname_divergingmap), bbox_inches='tight', pad_inches=0)

        mean_errors = compute_errors(gt_depth, pred_depth)
        print(("&{: 8.3}  " * 7).format(*mean_errors))

# --- Code cell 5 ---

fig, axes = plt.subplots(1, 1)
fig.set_size_inches(10, 6)
# axes.imshow(1-pred_depth, vmin=0, vmax=1)
im = axes.imshow(pred_depth - gt_depth, cmap="seismic_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=axes, aspect=12, drawedges=False, ticks=[-1, 0, 1])

```

## 文件: `eval2.py`

```py
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

# path of model
#model_name = "RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0"
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


cfgs = load_yaml("/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main/results/RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0/models/configs.yml")
cfgs.update({'device': device, "not_load_nets": ["net_pose_encoder", "net_pose_decoder","net_depth_intrinsics"]})



def getDepthNet():
    # cp_path = Path('results') / model_name / "models" / "weights_18"
    cp_path = Path('newresultsbest_bestcamera_attention_lightweightdepthdecoder') / model_name / "models" / "weights_23"
    # cp_path = Path('results') / model_name / "models" / "weights_last"

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
        gt_depth = cv2.imread(frame_fpath, -1)/(2**16)
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
    matcher_result_load = None #np.load(cfgs.get('matcher_result', None), allow_pickle=True).all()

    dataset = dataset(data_path, filenames, matcher_result_load,
        height, width, frame_ids, num_scales,
        is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

    model.set_eval()

    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(width, height))

    with torch.no_grad():
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].to(device)

            output = model.net_depth_decoder(model.net_depth_encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)
            # input_colors.append(data[("color", 0, 0)].numpy())
    pred_disps = np.concatenate(pred_disps)

    MIN_DEPTH = 0.001
    MAX_DEPTH = 1.
    errors = []
    ratios = []

    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
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
total_mean_errors = np.sum([mass * mean_errors for mass, mean_errors in zip(masses, test_mean_errors)], axis=0) / np.sum(masses)

print("\n  " + ("{:>8} | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "med", "std"))
print(("&{: 8.3}  " * 7).format(*total_mean_errors.tolist()) + "&{: 8.3}  &{: 8.3}  ".format(mean_med, mean_std) + "\\\\")
print("\n-> Done!")
```

## 文件: `eval2new.py`

```py
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
    #cp_path = Path('newresultsbest_bestcamera_attention_lightweightdepthdecoder') / model_name / "models" / "weights_23"

    #cp_path = Path('newresultsbest_bestcamera_attention') / model_name / "models" / "weights_23"
    # cp_path = Path('results') / model_name / "models" / "weights_last"
    cp_path = Path('newresults_0.15_0.15') / model_name / "models" / "weights_23"

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
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].to(device)

            output = model.net_depth_decoder(model.net_depth_encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)
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
```

## 文件: `eval3.py`

```py
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
```

## 文件: `eval5.py`

```py
# Converted from depth_color.ipynb
# Conversion time: 2025-10-22T07:31:44.132245
# This file was auto-generated by ChatGPT.
# Code cells are preserved. Markdown cells are added as commented blocks.

# --- Code cell 0 ---
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import os

os.chdir("/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main")

# imgs = ["data/c3vd_v2/imgs/cecum_t4_b_under_review/0000_color.png",
#         "data/c3vd_v2/imgs/desc_t4_a_under_review/0100_color.png",
#         "data/c3vd_v2/imgs/sigmoid_t3_b_under_review/0500_color.png",
#         "data/c3vd_v2/imgs/trans_t4_b_under_review/0400_color.png"]

imgs = ["data/c3vd_v2/imgs/cecum_t4_b_under_review/0028_color.png",
        "data/c3vd_v2/imgs/desc_t4_a_under_review/0020_color.png",
        "data/c3vd_v2/imgs/sigmoid_t3_b_under_review/0025_color.png",
        "data/c3vd_v2/imgs/trans_t4_b_under_review/0090_color.png"]

"""
imgs = [
    "data/c3vd_v2/imgs/cecum_t4_b_under_review/0060_color.png",
    # "data/c3vd_v2/imgs/cecum_t4_b_under_review/0000_color.png",
    "data/c3vd_v2/imgs/desc_t4_a_under_review/0030_color.png",
    # "data/c3vd_v2/imgs/desc_t4_a_under_review/0095_color.png",
    "data/c3vd_v2/imgs/sigmoid_t3_b_under_review/0120_color.png",
    # "data/c3vd_v2/imgs/sigmoid_t3_b_under_review/0000_color.png",
    "data/c3vd_v2/imgs/trans_t4_b_under_review/0455_color.png",
    # "data/c3vd_v2/imgs/trans_t4_b_under_review/0480_color.png",
]
"""
for img in imgs:
    fname = os.path.join(
        "/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main", img)
    gt = pil.open(fname.replace("color.png", "depth.tiff"))
    gt = np.array(gt, np.float32) / (2 ** 16 - 1)

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    im = axes.imshow(gt, cmap="jet_r", vmin=0, vmax=1)
    # axes.imshow(1-gt, vmin=0, vmax=1)
    axes.axis('off')
    fname_colormap = "_".join(img.split("/")[-2:] )
    #plt.savefig(os.path.join("gt_color_map", fname_colormap), bbox_inches='tight', pad_inches=0)
# axes[1].imshow(depths[idx].max() - depths[idx], cmap="jet")
# axes[1].axis('off')


# --- Code cell 1 ---
fig, axes = plt.subplots(1, 1)
fig.set_size_inches(10, 6)
im = axes.imshow(gt, cmap="jet_r", vmin=0, vmax=1)
# axes.imshow(1-gt, vmin=0, vmax=1)
axes.axis('off')
plt.colorbar(im, ax=axes, aspect=12, drawedges=False, ticks=[0, 1])

# --- Code cell 2 ---
import os

os.chdir("/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main")
from depthnet.utils import *
from depthnet.model import EstimateDepth
from pathlib import Path
import glob
import torchvision
from torchvision import transforms

model_list = ["RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2"]

#model_list = ["RCC"]
model_pred_depths = []


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


for model_name in model_list:
    # path of model
    # model_name = "RC_matching_depthnet_c3vd_v2_litemono_sd0"

    device = f'cuda:0'
    gpu_id = 0
    if gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    cfgs = load_yaml(
        "/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main/results/RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0/models/configs.yml")
    cfgs.update({'device': device, "not_load_nets": ["net_pose_encoder", "net_pose_decoder", "net_depth_intrinsics"]})


    def getDepthNet():
    # cp_path = Path('results') / model_name / "models" / "weights_18"
        #cp_path = Path('newresultsbest_bestcamera_attention_lightweightdepthdecoder') / model_name / "models" / "weights_23"
    # cp_path = Path极市results') / model_name / "models" / "weights_last"
        cp_path = Path('newresultsbest_bestcamera_attention') / model_name / "models" / "weights_23"
        #cp_path = Path('newresults_0.15_0.15') / model_name / "models" / "weights_22"
        model = globals().get(cfgs.get('model'))(cfgs)

        cp = {}
        for network_name in model.network_names:
            cp[network_name] = torch.load(cp_path / "{}.pth".format(network_name), map_location=device)

        model.load_model_state(cp)

        return model

    model = getDepthNet()
    model.to_device(device)

    # load
    import os
    import torch
    from torchvision import transforms
    import cv2
    import numpy as np
    from tqdm.auto import tqdm
    from PIL import Image

    from depthnet.networks.layers import disp_to_depth
    from depthnet.utils import readlines
    import depthnet.networks as networks
    from depthnet.networks.layers import transformation_from_parameters

    exps = [cfgs["model_name"]]

    height = cfgs.get('height', 256)
    width = cfgs.get('width', 320)
    model.set_eval()

    pred_depths = []

    print("-> Computing predictions with size {}x{}".format(width, height))

    MIN_DEPTH = 0.001
    MAX_DEPTH = 1.

    with torch.no_grad():
        for fname in tqdm(imgs):
            gt_depth = cv2.imread(fname.replace("color.png", "depth.tiff"), -1)
            gt_depth = gt_depth / (2 ** 16 - 1)
            gt_height, gt_width = gt_depth.shape[:2]

            input_color = pil_loader(fname)
            input_color = transforms.Resize((height, width), torchvision.transforms.InterpolationMode.LANCZOS)(
                input_color)

            # input_color2 = cv2.imread(fname)
            # input_color2 = cv2.resize(input_color2, (width, height))
            # rgb = rgb[:, :, ::-1]

            input_color = transforms.ToTensor()(input_color).unsqueeze(0).to(device)
            # input_color2 = transforms.ToTensor()(input_color2).unsqueeze(0).to(device)

            # input_color = cv2.imread(fname)
            # input_color = cv2.resize(input_color, (width, height))
            # # rgb = rgb[:, :, ::-1]
            # input_color = transforms.ToTensor()(input_color).unsqueeze(0).to(device)

            output = model.net_depth_decoder(model.net_depth_encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])

            pred_disp = pred_disp.cpu()[0, 0, :, :].numpy()
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            pred_depths.append(pred_depth)

    model_pred_depths.append(pred_depths)


# --- Code cell 3 ---
"""
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

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]
"""

fig, axes = plt.subplots(len(imgs), 8)
fig.set_size_inches(42, 20)
axes = axes.flatten()

rmse = np.zeros((len(imgs), 6))
"""
for i, model_name in enumerate(model_list):
    print(model_name)
    #print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    for j, img in enumerate(imgs):
        pred_depth = model_pred_depths[i][j]
        axes[8 * j + 2 + i].imshow(pred_depth, cmap="jet_r", vmin=0, vmax=1)
        axes[8 * j + 2 + i].axis('off')

        gt_depth = cv2.imread(img.replace("color.png", "depth.tiff"), -1)
        gt_depth = gt_depth / (2 ** 16 - 1)
        # axes.imshow(1-pred_depth, vmin=0, vmax=1)

        #mean_errors = compute_errors(gt_depth, pred_depth)
        print(("&{: 8.3}  " * 7).format(*mean_errors))

        # axes[18 * j + i].set_title("{:.3f}".format(mean_errors[2]), y=-0.14, size=36)
        #rmse[j, i] = mean_errors[2]
        # fontweight='bold'

        if i == 0:
            # input_color = cv2.imread(fname)
            # rgb = cv2.resize(input_color, (width, height))
            input_color = pil_loader(img)
            axes[8 * j + 0].imshow(input_color)
            axes[8 * j + 0].axis('off')

            # axes[8 * j + 1].imshow(gt_depth, cmap="plasma_r")
            axes[8 * j + 1].imshow(gt_depth, cmap="jet_r", vmin=0, vmax=1)
            axes[8 * j + 1].axis('off')

for j in range(len(imgs)):
    idx = np.argmin(rmse[j])
    for i in range(len(model_list)):
        if i == idx:
            axes[8 * j + 2 + i].set_title("{:.3f}".format(rmse[j, i]), y=-0.16, size=36, fontweight='bold')
        else:
            axes[8 * j + 2 + i].set_title("{:.3f}".format(rmse[j, i]), y=-0.16, size=36)

plt.tight_layout()
plt.show()
"""

"""
# --- Code cell 4 ---
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

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]
"""

for i, model_name in enumerate(model_list):
    print(model_name)
    #print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    for j, img in enumerate(imgs):
        pred_depth = model_pred_depths[i][j]
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)
        # axes.imshow(1-pred_depth, vmin=0, vmax=1)
        axes.imshow(pred_depth, cmap="jet_r", vmin=0, vmax=1)
        axes.axis('off')
        fname_colormap = "_".join(model_name.split("/")) + "_".join(img.split("/")[-2:] + ["colormap.png"])
        plt.savefig(os.path.join("pred_color_map", fname_colormap), bbox_inches='tight', pad_inches=0)

        gt_depth = cv2.imread(img.replace("color.png", "depth.tiff"), -1) / (2 ** 16 - 1)
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)
        # axes.imshow(1-pred_depth, vmin=0, vmax=1)
        axes.imshow(pred_depth - gt_depth, cmap="seismic_r", vmin=-1, vmax=1)
        axes.axis('off')
        fname_divergingmap = "_".join(model_name.split("/")) + "_".join(img.split("/")[-2:] + ["divergingmap.png"])
        plt.savefig(os.path.join("pred_diverging_map", fname_divergingmap), bbox_inches='tight', pad_inches=0)

        #mean_errors = compute_errors(gt_depth, pred_depth)
        #print(("&{: 8.3}  " * 7).format(*mean_errors))

# --- Code cell 5 ---

fig, axes = plt.subplots(1, 1)
fig.set_size_inches(10, 6)
# axes.imshow(1-pred_depth, vmin=0, vmax=1)
im = axes.imshow(pred_depth - gt_depth, cmap="seismic_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=axes, aspect=12, drawedges=False, ticks=[-1, 0, 1])
```

## 文件: `eval_new.py`

```py
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

save_dir = "./visual_depth_maps"
os.makedirs(save_dir, exist_ok=True)


def getDepthNet():
    # cp_path = Path('results') / model_name / "models" / "weights_18"
    cp_path = Path('newresults5') / model_name / "models" / "weights_23"
    # cp_path = Path('results') / model_name / "models" / "weights_last"

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
    def render_depth(disp):
        disp = (disp - disp.min()) / (disp.max() - disp.min()+1e-8)
        disp = np.power(disp,0.6)

        disp = (disp*255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
        return disp_color

    with torch.no_grad():
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].to(device)

            output = model.net_depth_decoder(model.net_depth_encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

            pred_disp = pred_disp[0]
            #pred_disp = cv2.resize(pred_disp, (775, 620))

            vis_pred_depth = render_depth(pred_disp)
            #raw_line = dataset.filenames[idx]
            #folder,_,frame_id,_ = raw_line.split()
            vis_file_name = os.path.join(save_dir, f"_depth.png")
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
        #print(gt_height) 1080
        #print(gt_width)1350

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
```

## 文件: `evaltime.py`

```py
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

# path of model
#model_name = "RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0"
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


cfgs = load_yaml("/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/hz/MonoLoT-main/results/RCC_matching_cropalign_depthnet_c3vd_v2_monodepth2_rep0/models/configs.yml")
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
        gt_depth = cv2.imread(frame_fpath, -1)/(2**16)
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
    matcher_result_load = None #np.load(cfgs.get('matcher_result', None), allow_pickle=True).all()

    dataset = dataset(data_path, filenames, matcher_result_load,
        height, width, frame_ids, num_scales,
        is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False)

    model.set_eval()

    pred_disps = []
    inference_times = []

    print("-> Computing predictions with size {}x{}".format(width, height))

    with torch.no_grad():
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].to(device)
            torch.cuda.synchronize() 
            start_time = time.time() 
            output = model.net_depth_decoder(model.net_depth_encoder(input_color))
            torch.cuda.synchronize()
            end_time = time.time() 
            inference_time = (end_time - start_time) * 1000
            inference_times.append(inference_time / input_color.shape[0]) 

            pred_disp, _ = disp_to_depth(output[("disp", 0)], cfgs["min_depth"], cfgs["max_depth"])
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)
            # input_colors.append(data[("color", 0, 0)].numpy())
    pred_disps = np.concatenate(pred_disps)
    avg_inference_time = np.mean(inference_times)
    print(f"\n-> 平均推理时间: {avg_inference_time:.3f} ms/张图像")

    MIN_DEPTH = 0.001
    MAX_DEPTH = 1.
    errors = []
    ratios = []

    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25     ).mean()
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
total_mean_errors = np.sum([mass * mean_errors for mass, mean_errors in zip(masses, test_mean_errors)], axis=0) / np.sum(masses)

print("\n  " + ("{:>8} | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "med", "std"))
print(("&{: 8.3}  " * 7).format(*total_mean_errors.tolist()) + "&{: 8.3}  &{: 8.3}  ".format(mean_med, mean_std) + "\\\\")
print("\n-> Done!")
```

## 文件: `evaltime1.py`

```py
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

```

## 文件: `huihua.py`

```py
from graphviz import Digraph

def draw_architecture():
    # 创建一个有向图，设置整体方向为从左到右 (LR)
    dot = Digraph('AlgorithmArchitecture', comment='Overall Framework of the Proposed Method')
    dot.attr(rankdir='LR', size='12,8', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='12', shape='rectangle', style='filled')

    # --- 第一部分：输入与 EAFE-M (创新点 4) ---
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='I. Pre-processing (EAFE-M Framework)', color='blue', fontname='Arial Bold')
        c.node('raw_img', 'Raw Endoscopic Image\n(I_t, I_s)', fillcolor='lightgray')
        
        # 创新点 4: EAFE-M
        c.node('eafe_m', 'Innovation 4: EAFE-M\n(Artifact Suppression & \nMask-guided Enhancement)', 
               fillcolor='#FFDAB9', penwidth='2') # 橙色背景
        
        c.node('mask', 'Binary Mask (M)', fillcolor='white', shape='note')
        c.node('enhanced_img', 'Enhanced Image (Ĩ)', fillcolor='white')

    # --- 第二部分：共享编码器 (创新点 1) ---
    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(label='II. Backbone', color='green', fontname='Arial Bold')
        # 创新点 1: EndoSHaTrans
        c.node('encoder', 'Innovation 1: EndoSHaTrans\n(Single-head Lightweight \nTransformer Encoder)', 
               fillcolor='#90EE90', penwidth='2') # 绿色背景

    # --- 第三部分：深度估计分支 (包含创新点 2) ---
    with dot.subgraph(name='cluster_depth') as c:
        c.attr(label='III. Depth Branch', color='red', fontname='Arial Bold')
        # 创新点 2: LDAM
        c.node('ldam', 'Innovation 2: LDAM\n(Two-stage Differential \nAttention Fusion)', 
               fillcolor='#FFB6C1', penwidth='2') # 粉色背景
        c.node('pixel_shuffle', 'PixelShuffle Decoder', fillcolor='white')
        c.node('depth_out', 'Pixel-wise Depth Map (D)', fillcolor='white', shape='parallelogram')

    # --- 第四部分：位姿-内参分支 (创新点 3) ---
    with dot.subgraph(name='cluster_pose') as c:
        c.attr(label='IV. Pose-Intrinsic Branch', color='purple', fontname='Arial Bold')
        # 创新点 3: EndoMSID
        c.node('msid', 'Innovation 3: EndoMSID\n(Context-aware Joint \nDecoder)', 
               fillcolor='#E6E6FA', penwidth='2') # 紫色背景
        c.node('pose_out', '6-DoF Pose (T)', fillcolor='white', shape='parallelogram')
        c.node('intri_out', 'Camera Intrinsics (K)', fillcolor='white', shape='parallelogram')

    # --- 第五部分：联合损失监督 ---
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='V. Self-supervised Loss', color='darkorange', fontname='Arial Bold')
        c.node('warping', 'View Synthesis (Warping)', fillcolor='lightyellow')
        c.node('total_loss', 'Joint Loss Function\n(L_repro, L_smooth, L_hard)', fillcolor='orange')

    # --- 连接所有逻辑线 ---
    # 输入 -> EAFE-M
    dot.edge('raw_img', 'eafe_m')
    dot.edge('eafe_m', 'mask')
    dot.edge('eafe_m', 'enhanced_img')
    
    # 增强图 -> 编码器
    dot.edge('enhanced_img', 'encoder')
    
    # 编码器 -> 深度分支 (经过 LDAM)
    dot.edge('encoder', 'ldam', label='Features (F1-F5)')
    dot.edge('ldam', 'pixel_shuffle')
    dot.edge('pixel_shuffle', 'depth_out')
    
    # 编码器 -> 位姿分支
    dot.edge('encoder', 'msid', label='High-order Feat.')
    dot.edge('msid', 'pose_out')
    dot.edge('msid', 'intri_out')
    
    # 最终汇总到 Loss
    dot.edge('depth_out', 'warping')
    dot.edge('pose_out', 'warping')
    dot.edge('intri_out', 'warping')
    dot.edge('raw_img', 'warping', style='dashed', label='Raw Target Image')
    
    dot.edge('warping', 'total_loss')
    dot.edge('mask', 'total_loss', color='red', style='dashed', label='Masking')

    # 保存并查看
    dot.render('my_algorithm_architecture', format='png', view=True)
    print("架构图已生成并保存为 my_algorithm_architecture.png")

if __name__ == "__main__":
    draw_architecture()
```

## 文件: `plot_test.py`

```py
from depthnet.meters import plot_metrics
import json

json_path = r"e:\CodeWork\PyWork\MonoLoT\results\<exp_name>\metrics.json"
out_pdf   = r"e:\CodeWork\PyWork\MonoLoT\results\<exp_name>\metrics.pdf"

with open(json_path, "r") as f:
    data = json.load(f)

plot_metrics(data, pdf_path=out_pdf)
```

## 文件: `reconstruction.py`

```py
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

```

## 文件: `run.py`

```py
import argparse
import json

from depthnet import setup_runtime, Trainer
from depthnet.model import EstimateDepth

import torch
import os


if __name__ == "__main__":
    os.system("nvidia-smi")

    ## runtime arguments
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', default="/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/experiments/c3vd_v2/monodepth2/baseline_c3vd_v2_monodepth2-wjy.yml", type=str, help='Specify a config file path')
    parser.add_argument('--gpu', default=torch.device("cuda"), type=int, help='Specify a GPU device')
    parser.add_argument('--gpu_any', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
    parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
    parser.add_argument('--cfg_params', default='{}', type=str, help='Manually add entries to the config dict')
    args = parser.parse_args()

    ## set up
    cfgs = setup_runtime(args)
    print(args.cfg_params)
    cfg_params = json.loads(args.cfg_params)
    cfgs.update(cfg_params)

    if 'model' in cfgs:
        model = globals().get(cfgs.get('model'))
    else:
        model = EstimateDepth
    trainer = Trainer(cfgs, model)
    run_train = cfgs.get('run_train', False)
    # run_test = cfgs.get('run_test', False)

    ## run
    if run_train:
        trainer.train()
    # if run_test:
    #     trainer.test()

```

## 文件: `run_custom.py`

```py
import argparse
import json

from depthnet import setup_runtime, Trainer
from depthnet.model import EstimateDepth

import torch
import os


if __name__ == "__main__":
    os.system("nvidia-smi")

    ## runtime arguments
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', default="/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/experiments/c3vd_v2/monodepth2/baseline_c3vd_v2_monodepth2-wjy.yml", type=str, help='Specify a config file path')
    parser.add_argument('--gpu', default=torch.device("cuda"), type=int, help='Specify a GPU device')
    parser.add_argument('--gpu_any', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
    parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
    parser.add_argument('--cfg_params', default='{}', type=str, help='Manually add entries to the config dict')
    args = parser.parse_args()

    ## set up
    cfgs = setup_runtime(args)
    print(args.cfg_params)
    cfg_params = json.loads(args.cfg_params)
    cfgs.update(cfg_params)

    if 'model' in cfgs:
        model = globals().get(cfgs.get('model'))
    else:
        model = EstimateDepth
    trainer = Trainer(cfgs, model)
    run_train = cfgs.get('run_train', False)
    # run_test = cfgs.get('run_test', False)

    ## run
    if run_train:
        trainer.train()
    # if run_test:
    #     trainer.test()

```

## 文件: `test.py`

```py

```

## 目录: `depthnet`

### 文件: `__init__.py`

```py
from .utils import setup_runtime
from .trainer import Trainer
from .model import EstimateDepth
"""
from .model_sc import EstimateDepthSC
"""
```

### 文件: `kitti_utils.py`

```py
from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

```

### 文件: `lr_scheduler.py`

```py
import math
import torch
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpScheduler(_LRScheduler):
    """
    Args:
        optimizer: [torch.optim.Optimizer] only pass if using as astand alone lr_scheduler
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        eta_min: float = 0.0,
        last_epoch=-1,
        max_lr: Optional[float] = 0.1,
        warmup_steps: Optional[int] = 0,
    ):

        if warmup_steps != 0:
            assert warmup_steps >= 0

        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.step_in_cycle = last_epoch
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps  # warmup

        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]

        else:
            return [base_lr + (self.max_lr - base_lr) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

        else:
            self.step_in_cycle = self.epoch

        self.max_lr = self.base_max_lr
        self.last_epoch = math.floor(self.epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnealingWarmRestartsWeightDecay(_LRScheduler):
    """
       Helper class for chained scheduler not to used directly. this class is synchronised with
       previous stage i.e.  WarmUpScheduler (max_lr, T_0, T_cur etc) and is responsible for
       CosineAnealingWarmRestarts with weight decay
       """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mul: float = 1.,
        eta_min: float = 0.001,
        last_epoch=-1,
        max_lr: Optional[float] = 0.1,
        gamma: Optional[float] = 1.,
    ):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_i = T_0  # number of epochs between two warm restarts
        self.cycle = 0
        self.eta_min = eta_min
        self.gamma = gamma
        self.T_cur = last_epoch  # number of epochs since the last restart
        super(CosineAnealingWarmRestartsWeightDecay, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        return [
            base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mul

        # since warmup steps must be < T_0 and if epoch count > T_0 we just apply cycle count for weight decay
        if self.epoch >= self.T_0:
            if self.T_mul == 1.:
                self.T_cur = self.epoch % self.T_0
                self.cycle = self.epoch // self.T_0
            else:
                n = int(math.log((self.epoch / self.T_0 * (self.T_mul - 1) + 1), self.T_mul))
                self.cycle = n
                self.T_cur = self.epoch - int(self.T_0 * (self.T_mul**n - 1) / (self.T_mul - 1))
                self.T_i = self.T_0 * self.T_mul**(n)

        # base condition that applies original implementation for cosine cycles for details visit:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        else:
            self.T_i = self.T_0
            self.T_cur = self.epoch

        # this is where weight decay is applied
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(self.epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ChainedScheduler(_LRScheduler):
    """
    Driver class
        Args:
        T_0: First cycle step size, Number of iterations for the first restart.
        T_mul: multiplicative factor Default: -1., A factor increases T_i after a restart
        eta_min: Min learning rate. Default: 0.001.
        max_lr: warmup's max learning rate. Default: 0.1. shared between both schedulers
        warmup_steps: Linear warmup step size. Number of iterations to complete the warmup
        gamma: Decrease rate of max learning rate by cycle. Default: 1.0 i.e. no decay
        last_epoch: The index of last epoch. Default: -1

    Usage:

        ChainedScheduler without initial warmup and weight decay:

            scheduler = ChainedScheduler(
                            optimizer,
                            T_0=20,
                            T_mul=2,
                            eta_min = 1e-5,
                            warmup_steps=0,
                            gamma = 1.0
                        )

        ChainedScheduler with weight decay only:
            scheduler = ChainedScheduler(
                            self,
                            optimizer: torch.optim.Optimizer,
                            T_0: int,
                            T_mul: float = 1.0,
                            eta_min: float = 0.001,
                            last_epoch=-1,
                            max_lr: Optional[float] = 1.0,
                            warmup_steps: int = 0,
                            gamma: Optional[float] = 0.9
                        )

        ChainedScheduler with initial warm up and weight decay:
            scheduler = ChainedScheduler(
                            self,
                            optimizer: torch.optim.Optimizer,
                            T_0: int,
                            T_mul: float = 1.0,
                            eta_min: float = 0.001,
                            last_epoch = -1,
                            max_lr: Optional[float] = 1.0,
                            warmup_steps: int = 10,
                            gamma: Optional[float] = 0.9
                        )
    Example:
        >>> model = AlexNet(num_classes=2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-1)
        >>> scheduler = ChainedScheduler(
        >>>                 optimizer,
        >>>                 T_0 = 20,
        >>>                 T_mul = 1,
        >>>                 eta_min = 0.0,
        >>>                 gamma = 0.9,
        >>>                 max_lr = 1.0,
        >>>                 warmup_steps= 5 ,
        >>>             )
        >>> for epoch in range(100):
        >>>     optimizer.step()
        >>>     scheduler.step()

    Proper Usage:
        https://wandb.ai/wandb_fc/tips/reports/How-to-Properly-Use-PyTorch-s-CosineAnnealingWarmRestarts-Scheduler--VmlldzoyMTA3MjM2

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mul: float = 1.0,
        eta_min: float = 0.001,
        last_epoch=-1,
        max_lr: Optional[float] = 1.0,
        warmup_steps: Optional[int] = 5,
        gamma: Optional[float] = 0.95,
    ):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        if warmup_steps != 0:
            assert warmup_steps < T_0
            warmup_steps = warmup_steps + 1  # directly refers to epoch account for 0 off set

        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_i = T_0  # number of epochs between two warm restarts
        self.cycle = 0
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps  # warmup
        self.gamma = gamma
        self.T_cur = last_epoch  # number of epochs since the last restart
        self.last_epoch = last_epoch

        self.cosine_scheduler1 = WarmUpScheduler(
            optimizer,
            eta_min=self.eta_min,
            warmup_steps=self.warmup_steps,
            max_lr=self.max_lr,
        )
        self.cosine_scheduler2 = CosineAnealingWarmRestartsWeightDecay(
            optimizer,
            T_0=self.T_0,
            T_mul=self.T_mul,
            eta_min=self.eta_min,
            max_lr=self.max_lr,
            gamma=self.gamma,
        )

    def get_lr(self):
        if self.warmup_steps != 0:
            if self.epoch < self.warmup_steps:
                return self.cosine_scheduler1.get_lr()
        if self.epoch >= self.warmup_steps:
            return self.cosine_scheduler2.get_lr()

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1

        if self.warmup_steps != 0:
            if self.epoch < self.warmup_steps:
                self.cosine_scheduler1.step()
                self.last_epoch = self.epoch

        if self.epoch >= self.warmup_steps:
            self.cosine_scheduler2.step()
            self.last_epoch = self.epoch

```

### 文件: `meters.py`

```py
import collections
import json
import os
import time

import matplotlib.pyplot as plt
import torch

from .utils import xmkdir


class TotalAverage():
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_value = 0.
        self.mass = 0.
        self.sum = 0.

    def update(self, value, mass=1):
        self.last_value = value
        self.mass += mass
        self.sum += value * mass

    def get(self):
        return self.sum / self.mass


class MovingAverage():
    def __init__(self, inertia=0.9):
        self.inertia = inertia
        self.reset()
        self.last_value = None

    def reset(self):
        self.last_value = None
        self.average = None

    def update(self, value, mass=1):
        self.last_value = value
        if self.average is None:
            self.average = value
        else:
            self.average = self.inertia * self.average + (1 - self.inertia) * value

    def get(self):
        return self.average


class MetricsTrace():
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {}

    def append(self, dataset, metric):
        if dataset not in self.data:
            self.data[dataset] = []
        self.data[dataset].append(metric.get_data_dict())

    def load(self, path):
        """Load the metrics trace from the specified JSON file."""
        with open(path, 'r') as f:
            self.data = json.load(f)

    def save(self, path):
        """Save the metrics trace to the specified JSON file."""
        if path is None:
            return
        xmkdir(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def plot(self, pdf_path=None):
        """Plots and optionally save as PDF the metrics trace."""
        plot_metrics(self.data, pdf_path=pdf_path)

    def get(self):
        return self.data

    def __str__(self):
        pass


class Metrics():
    def __init__(self):
        self.iteration_time = MovingAverage(inertia=0.9)
        self.now = time.time()

    def update(self, prediction=None, ground_truth=None):
        self.iteration_time.update(time.time() - self.now)
        self.now = time.time()

    def get_data_dict(self):
        return {"objective" : self.objective.get(), "iteration_time" : self.iteration_time.get()}


class StandardMetrics(Metrics):
    def __init__(self, m=None, metric_str_exclude=[]):
        super(StandardMetrics, self).__init__()
        self.metrics = m or {}
        self.speed = MovingAverage(inertia=0.9)
        self.metric_str_exclude = metric_str_exclude

    def update(self, metric_dict, mass=1):
        super(StandardMetrics, self).update()
        for metric, val in metric_dict.items():
            if torch.is_tensor(val):
                val = val.item()
            if metric not in self.metrics:
                self.metrics[metric] = TotalAverage()
            self.metrics[metric].update(val, mass)
        self.speed.update(mass / self.iteration_time.last_value)

    def get_data_dict(self):
        data_dict = {k: v.get() for k,v in self.metrics.items()}
        data_dict['speed'] = self.speed.get()
        return data_dict

    def __str__(self):
        pstr = '%7.1fHz\t' %self.speed.get()
        pstr += '\t'.join(['%s: %6.5f' %(k,v.get()) for k,v in self.metrics.items() if k not in self.metric_str_exclude])
        return pstr


def plot_metrics(stats, pdf_path=None, fig=1, datasets=None, metrics=None):
    """Plot metrics. `stats` should be a dictionary of type

          stats[dataset][t][metric][i]

    where dataset is the dataset name (e.g. `train` or `val`), t is an iteration number,
    metric is the name of a metric (e.g. `loss` or `top1`),  and i is a loss dimension.

    Alternatively, if a loss has a single dimension, `stats[dataset][t][metric]` can
    be a scalar.

    The supported options are:

    - pdf_file: path to a PDF file to store the figure (default: None)
    - fig: MatPlotLib figure index (default: 1)
    - datasets: list of dataset names to plot (default: None)
    - metrics: list of metrics to plot (default: None)
    """
    plt.figure(fig, figsize=[8, 11])
    plt.clf()
    linestyles = ['-', '--', '-.', ':']
    datasets = list(stats.keys()) if datasets is None else datasets
    # Filter out empty datasets
    datasets = [d for d in datasets if len(stats[d]) > 0]
    duration = len(stats[datasets[0]])
    metrics = list(stats[datasets[0]][0].keys()) if metrics is None else metrics
    metric_str_exclude = ["loss/photometric_1", "loss/photometric_2", "loss/photometric_3", "de/abs_rel", "de/sq_rel", "de/rms", "da/a2", "da/a3"]
    metrics = list(metric for metric in metrics if metric not in metric_str_exclude)
        
    for m, metric in enumerate(metrics):
        plt.subplot(len(metrics),1,m+1)
        legend_content = []
        for d, dataset in enumerate(datasets):
            ls = linestyles[d % len(linestyles)]
            if isinstance(stats[dataset][0][metric], collections.Iterable):
                metric_dimension = len(stats[dataset][0][metric])
                for sl in range(metric_dimension):
                    x = [stats[dataset][t][metric][sl] for t in range(duration)]
                    plt.plot(x, linestyle=ls)
                    name = f'{dataset} {metric}[{sl}]'
                    legend_content.append(name)
            else:
                x = [stats[dataset][t][metric] for t in range(duration)]
                plt.plot(x, linestyle=ls)
                name = f'{dataset} {metric}'
                legend_content.append(name)
        plt.legend(legend_content, loc=(1.04,0))
        plt.grid(True)
    if pdf_path is not None:
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.draw()
    plt.pause(0.0001)

```

### 文件: `model.py`

```py
from .networks import *
from .lr_scheduler import ChainedScheduler
import os
from .networks.layers import *
import random
from .networks.pose_decoder import PoseDecoder
from .networks.resnet_encoder import ResnetEncoder
from .networks.depth_decoder import DepthDecoder
#from .networks.cednet import CEDNet
#from .networks.ghostnetv3 import ghostnetv3
from .networks.swiftformer import SwiftFormer_S
#from .networks.depth_encoder import LiteMono
#from .networks.depth_decoder_litemono import DepthDecoderV2
#from .networks.pose_decode_litemono import PoseDecoderV2
from .networks.intrinsics_decoder import IntrinsicsHead

from .networks.shvit import SHViTEncoder

EPS = 1e-7


class EstimateDepth():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.height = cfgs.get('height', 256)
        self.width = cfgs.get('width', 320)
        self.batch_size = cfgs.get('batch_size', 64)

        # checking height and width are multiples of 32
        assert self.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = cfgs.get('device', 'cpu')
        self.scales = cfgs.get('scales', [0, 1, 2, 3])
        self.num_scales = len(self.scales)
        self.frame_ids = cfgs.get('frame_ids', [0, -1, 1])
        self.num_pose_frames = 2
        self.disable_automasking = cfgs.get('disable_automasking', False)

        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

        # depth
        self.min_depth = cfgs.get('min_depth', 0.1)
        self.max_depth = cfgs.get('max_depth', 100.0)
        self.min_gt_depth = cfgs.get('min_gt_depth', 0.001)
        self.max_gt_depth = cfgs.get('max_gt_depth', 1.)

        self.model_str = cfgs.get('model_str', 'monodepth2')
        self.scheduler_str = cfgs.get('scheduler_str', None)
        self.num_layers = cfgs.get('num_layers', 18)
        self.weights_init = cfgs.get('weights_init', "pretrained")
        if self.model_str == "monodepth2":
            # self.net_depth_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained")
            # self.net_depth_encoder = ghostnetv3(width=1.0)
            self.net_depth_encoder = SwiftFormer_S()
            self.net_depth_decoder = DepthDecoder(np.array([24, 48, 64, 168, 224]), [0, 1, 2, 3], )

            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained",
                                                  num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1,
                                                num_frames_to_predict_for=2, use_contmix=True)

            self.net_depth_intrinsics = IntrinsicsHead(self.net_pose_encoder.num_ch_enc, use_contmix=True)
        elif self.model_str == "shvit":
            # 1. 初始化编码器
            self.net_depth_encoder = SHViTEncoder(
                model_type='shvit_s1',
                height=self.height,
                width=self.width
            )

            # 【显式加载】直接在这里写死你的权重路径
            # 你可以将这里的字符串替换为你电脑上 shvit_s1.pth 的真实绝对路径
            shvit_weight_path = "/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/shvit_s1.pth"
            self.net_depth_encoder.load_pretrained(shvit_weight_path)

            # 2. 初始化解码器
            # 注意：SHViT 只提供 3 个尺度的特征 (1/16, 1/32, 1/64)
            # 原有的 DepthDecoder 默认适配 ResNet 的 5 个尺度
            # 建议使用 LiteMono 的 DepthDecoderV2，它能完美处理 3 个尺度的输入
            # from .networks.depth_decoder_litemono import DepthDecoderV2
            # self.net_depth_decoder = DepthDecoderV2(
            #     self.net_depth_encoder.num_ch_enc,
            #     self.scales
            # )
            # 不要使用 DepthDecoderV2，因为它默认只有 3 级
            from .networks.depth_decoder import DepthDecoder
            self.net_depth_decoder = DepthDecoder(
                self.net_depth_encoder.num_ch_enc,
                self.scales
            )

            # 3. Pose 部分 (维持原样，通常用 ResNet 比较稳)
            self.net_pose_encoder = ResnetEncoder(
                self.num_layers,
                self.weights_init == "pretrained",
                num_input_images=self.num_pose_frames
            )
            self.net_pose_decoder = PoseDecoder(
                self.net_pose_encoder.num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
                use_contmix=True
            )
            self.net_depth_intrinsics = IntrinsicsHead(self.net_pose_encoder.num_ch_enc, use_contmix=True)
        elif self.model_str in ["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"]:
            self.drop_path = cfgs.get('drop_path', 0.2)
            self.net_depth_encoder = LiteMono(model=self.model_str, drop_path_rate=self.drop_path, width=self.width,
                                              height=self.height)
            self.net_depth_decoder = DepthDecoderV2(self.net_depth_encoder.num_ch_enc, self.scales)
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained",
                                                  num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoderV2(self.net_pose_encoder.num_ch_enc, num_input_features=1,
                                                  num_frames_to_predict_for=2)
        elif self.model_str == 'monovit':
            self.net_depth_encoder = mpvit_small()
            self.net_depth_decoder = MonovitDecoder()
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained",
                                                  num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1,
                                                num_frames_to_predict_for=2)
        else:
            raise NotImplementedError

        # optim
        self.start_epoch = cfgs.get('start_epoch', 0)
        self.num_epochs = cfgs.get('num_epochs', 20)
        self.lr = cfgs.get('lr', [0.0001, 5e-6, 36, 0.0001, 1e-5, 36])
        self.weight_decay = cfgs.get('weight_decay', 0.02)
        self.disparity_smoothness = cfgs.get('disparity_smoothness', 0.001)

        # matcher loss
        self.disable_matcher = cfgs.get('disable_matcher', False)
        self.confidence = cfgs.get('confidence', 0.9)
        self.matcher_loss_alpha = cfgs.get('matcher_loss_alpha', 0.2)
        self.half_epoch_matcher = cfgs.get('half_epoch_matcher', False)
        self.matcher_loss_delta = cfgs.get('matcher_loss_delta', 0)

        self.network_names = [k for k in vars(self) if k.startswith('net')]
        # optimizer
        if self.model_str in ['monovit', 'lite-mono', 'shvit']:  # 增加 shvit
            self.make_optimizer = lambda optim_dict: torch.optim.AdamW(
                optim_dict["parameters"], lr=optim_dict["lr"], weight_decay=self.weight_decay)
        elif self.model_str == "monodepth2":

            self.make_optimizer = lambda optim_dict: torch.optim.Adam(
                optim_dict["parameters"], lr=optim_dict["lr"])
            """
            self.make_optimizer = lambda optim_dict: torch.optim.AdamW(
                optim_dict["parameters"], lr=optim_dict["lr"], betas=(0.9, 0.999), weight_decay=self.weight_decay)
            """

        # ratio consistency
        self.ratio_consistency = cfgs.get('ratio_consistency', False)
        self.ratio_consistency_crop = cfgs.get('ratio_consistency_crop', False)
        self.ratio_consistency_normalization = cfgs.get('ratio_consistency_normalization', False)
        self.ratio_consistency_scales_normalization = cfgs.get('ratio_consistency_scales_normalization', False)
        self.weight_ratio_consistency_crop = cfgs.get('weight_ratio_consistency_crop', 1.0)
        self.align_crop_position = cfgs.get('align_crop_position', False)

        # geometry loss
        self.geometry_loss = cfgs.get('geometry_loss', False)
        self.geometry_loss_disp_mode = cfgs.get('geometry_loss_disp_mode', False)

        # load
        self.load_weights_folder = cfgs.get('load_weights_folder', None)
        self.mypretrain = cfgs.get('mypretrain', None)
        self.not_load_nets = cfgs.get('not_load_nets', ())
        self.not_load_optimizer = cfgs.get('not_load_optimizer', ())
        self.models_to_load = cfgs.get('models_to_load', [])

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        # no grad layers
        self.ssim = SSIM()
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)
            setattr(self, "backproject_depth_{}".format(scale), BackprojectDepth(self.batch_size, h, w))
            setattr(self, "project_3d_{}".format(scale), Project3D(self.batch_size, h, w))

        self.other_param_names = ['ssim']
        for scale in self.scales:
            self.other_param_names += ["backproject_depth_{}".format(scale), "project_3d_{}".format(scale)]

    def init_optimizers(self):
        # optim
        self.optimizer_names = []
        self.parameters_depth = []
        self.parameters_pose = []
        for net_name in self.network_names:
            if not any([p.requires_grad for p in getattr(self, net_name).parameters()]):
                continue
            if net_name.startswith('net_depth'):
                self.parameters_depth += list(getattr(self, net_name).parameters())
            elif net_name.startswith('net_pose'):
                self.parameters_pose += list(getattr(self, net_name).parameters())
        self.optimizer_depth = self.make_optimizer({"parameters": self.parameters_depth, "lr": self.lr[0]})
        self.optimizer_pose = self.make_optimizer({"parameters": self.parameters_pose, "lr": self.lr[3]})
        self.optimizer_names += ["optimizer_depth", "optimizer_pose"]

        # scheduler
        self.scheduler_names = []
        if self.scheduler_str is not None:
            print("scheduler_str mode...")
            if self.scheduler_str == 'cosine':
                self.scheduler_depth_lr = ChainedScheduler(
                    self.optimizer_depth,
                    T_0=int(self.lr[2]), T_mul=1, eta_min=self.lr[1], last_epoch=self.start_epoch - 1,
                    max_lr=self.lr[0], warmup_steps=0, gamma=0.9)
                self.scheduler_pose_lr = ChainedScheduler(
                    self.optimizer_pose,
                    T_0=int(self.lr[5]), T_mul=1, eta_min=self.lr[4], last_epoch=self.start_epoch - 1,
                    max_lr=self.lr[3], warmup_steps=0, gamma=0.9)
                print("optimiser cosine")
            elif self.scheduler_str == 'exp':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_depth, 0.9)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pose, 0.9)
                print("optimiser exp")
            elif self.scheduler_str == 'step':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_depth, self.num_epochs - 5,
                                                                          0.1)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_pose, self.num_epochs - 5, 0.1)
                print("optimiser step")
        else:
            if self.model_str == 'lite-mono' or self.model_str == 'shvit': # <--- 修改这里，加上 'shvit':
                self.scheduler_depth_lr = ChainedScheduler(
                    self.optimizer_depth,
                    T_0=int(self.lr[2]), T_mul=1, eta_min=self.lr[1], last_epoch=self.start_epoch - 1,
                    max_lr=self.lr[0], warmup_steps=0, gamma=0.9)
                self.scheduler_pose_lr = ChainedScheduler(
                    self.optimizer_pose,
                    T_0=int(self.lr[5]), T_mul=1, eta_min=self.lr[4], last_epoch=self.start_epoch - 1,
                    max_lr=self.lr[3], warmup_steps=0, gamma=0.9)
                print("optimiser cosine")
            elif self.model_str == 'monovit':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_depth, 0.9)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pose, 0.9)
                print("optimiser exp")
            elif self.model_str == 'monodepth2':

                self.scheduler_depth_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_depth, self.num_epochs - 5,
                                                                          0.1)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_pose, self.num_epochs - 5, 0.1)
                """
                self.scheduler_depth_lr = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_depth,
                                                                               [15], 0.1)
                self.scheduler_pose_lr = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pose, [15],

                                                                           0.1)
                """

                print("optimiser step")

        self.scheduler_names += ["scheduler_depth_lr", "scheduler_pose_lr"]

    def load_model(self):
        """Load model(s) from disk
        """
        self.load_weights_folder = os.path.expanduser(self.load_weights_folder)

        assert os.path.isdir(self.load_weights_folder), \
            "Cannot find folder {}".format(self.load_weights_folder)
        print("loading model from folder {}".format(self.load_weights_folder))

        for n in self.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.load_weights_folder, "{}.pth".format(n))

            model_dict = getattr(self, n).state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            getattr(self, n).load_state_dict(model_dict)

        # loading optimizer state
        optimizer_depth_load_path = os.path.join(self.load_weights_folder, "optimizer_depth.pth")
        optimizer_pose_load_path = os.path.join(self.load_weights_folder, "optimizer_pose.pth")
        if os.path.isfile(optimizer_depth_load_path) and os.path.isfile(optimizer_pose_load_path):
            print("Loading optimizer weights")
            self.optimizer_depth.load_state_dict(torch.load(optimizer_depth_load_path))
            self.optimizer_pose.load_state_dict(torch.load(optimizer_pose_load_path))
        else:
            print("Cannot find optimizer weights so Adam is randomly initialized")

    def load_shvit_weights(self, path):
        """为 SHViT-S1 加载预训练权重"""
        if not path or not os.path.isfile(path):
            print(f"=> [Error] Pretrain file not found at: {path}")
            return

        print(f"=> Loading SHViT-S1 weights from: {path}")
        checkpoint = torch.load(path, map_location="cpu")

        # 提取 state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 清理键名：
        # 1. 移除 'backbone.' 或 'module.' 前缀
        # 2. 忽略分类头 'head.' 或 'fc.'
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[7:]
            if name.startswith('backbone.'): name = name[9:]

            # 过滤掉不属于编码器的层（分类层）
            if any(x in name for x in ['head', 'fc', 'classifier']):
                continue

            new_state_dict[name] = v

        # 加载到 net_depth_encoder.encoder 中
        # 因为我们的封装结构是：SHViTEncoder -> self.encoder (Partial_ViT_Exp)
        msg = self.net_depth_encoder.encoder.load_state_dict(new_state_dict, strict=False)

        print(f"=> Successfully loaded weights.")
        print(f"=> Missing keys (should only be head-related): {msg.missing_keys[:5]}")

    def load_pretrain(self):

        # 检查是否是 SHViT 模型
        if self.model_str == 'shvit':
            self.load_shvit_weights(self.mypretrain)
            return  # 加载完直接返回

        # 以下是原有的加载逻辑 (针对 lite-mono 等)
        if self.mypretrain is None:
            return

        # only designed for lite-mono
        # self.mypretrain = os.path.expanduser(self.mypretrain)
        path = self.mypretrain
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        state_dict = _state_dict
        self.net_depth_encoder.load_state_dict(state_dict, strict=False)
        print('mypretrain loaded.')

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names and k not in self.not_load_nets:
                print("Loading ", k)
                model_dict = getattr(self, k).state_dict()
                getattr(self, k).load_state_dict({k: v for k, v in cp[k].items() if k in model_dict})

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names and k not in self.not_load_optimizer:
                print("Loading ", k)
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))

        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def backward(self, losses):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        losses["loss"].backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def forward(self, inputs):
        """Feedforward once."""
        for key, ipt in inputs.items():
            if "correspondences" in key:
                inputs[key] = ipt
            else:
                inputs[key] = ipt.to(self.device)

        # we only feed the image with frame_id 0 through the depth encoder
        features = self.net_depth_encoder(inputs["color_aug", 0, 0])

        outputs = self.net_depth_decoder(features)

        x = self.net_depth_encoder(inputs[("color_local_aug", 0, 0)])
        o = self.net_depth_decoder(x)
        for i in range(1):
            outputs[("disp_local", i)] = o[("disp", i)]

        x = self.net_depth_encoder(inputs[("color_reshuffle_aug", 0, 0)])
        o = self.net_depth_decoder(x)
        for i in range(1):
            outputs[("disp_reshuffle", i)] = o[("disp", i)]

            all_disp = []
            for b in range(self.batch_size):
                ### Split-Permute as depicted in paper (vertical + horizontal)
                split_x = inputs[("split_xy")][b][0].item()
                split_y = inputs[("split_xy")][b][1].item()
                split_x = round(split_x / (2 ** i))
                split_y = round(split_y / (2 ** i))
                disp_reshuffle = outputs[("disp_reshuffle", i)][b]  # 1*H*W
                patch1 = disp_reshuffle[:, 0:split_y, :]
                patch2 = disp_reshuffle[:, split_y:, :]
                disp_restore = torch.cat([patch2, patch1], dim=1)
                patch1 = disp_restore[:, :, 0:split_x]
                patch2 = disp_restore[:, :, split_x:]
                disp_restore = torch.cat([patch2, patch1], dim=2)
                all_disp.append(disp_restore)

                ### Split-Permute (vertical or horizontal, randomly choose one)
                # split_x = inputs[("split_xy", i)][b][0].item()
                # split_y = inputs[("split_xy", i)][b][1].item()
                # split_x = round(split_x / (2 ** i))
                # split_y = round(split_y / (2 ** i))
                # disp_reshuffle = outputs[("disp_reshuffle", i)][b]   #1*H*W
                # if split_x == 0:
                #     patch1 = disp_reshuffle[:, 0:split_y, :]
                #     patch2 = disp_reshuffle[:, split_y:, :]
                #     disp_restore = torch.cat([patch2, patch1], dim=1)
                # else:
                #     patch1 = disp_reshuffle[:, :, 0:split_x]
                #     patch2 = disp_reshuffle[:, :, split_x:]
                #     disp_restore = torch.cat([patch2, patch1], dim=2)
                # all_disp.append(disp_restore)
            disp_restore = torch.stack(all_disp, dim=0)
            outputs[("disp_reshuffle", i)] = disp_restore

        outputs.update(self.predict_poses(inputs))

        # depth in frame [-1, 1] and scales [0, 1, 2, 3]
        if self.geometry_loss:
            for i, frame_id in enumerate(self.frame_ids[1:]):
                outputs_disp_scales = self.net_depth_decoder(self.net_depth_encoder(inputs["color_aug", frame_id, 0]))
                for scale in self.scales:
                    disp = outputs_disp_scales[("disp", scale)]
                    disp = F.interpolate(
                        disp, [self.height, self.width], mode="bilinear", align_corners=False)
                    _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
                    outputs[("depth", frame_id, scale)] = depth
                    if self.geometry_loss_disp_mode:
                        outputs[("disp", frame_id, scale)] = disp

        # shuffle
        outputs["do_shuffle"] = random.random() > 0.5
        if self.ratio_consistency and outputs["do_shuffle"]:
            direction = random.random() > 0.5
            inputs["shuffle_color_aug", 0, 0] = self.layer_shuffle(inputs["color_aug", 0, 0], direction)
            shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
            shuffle_outputs = self.net_depth_decoder(shuffle_features)
            for scale in self.scales:
                outputs[("shuffle_disp", scale)] = self.layer_shuffle(shuffle_outputs[("disp", scale)], direction)
        elif self.ratio_consistency_crop and outputs["do_shuffle"]:
            b, _, h, w = inputs["color_aug", 0, 0].shape
            crop_info = self.get_crop_info(h // 2 ** (self.num_scales - 1), w // 2 ** (self.num_scales - 1),
                                           align_crop_position=self.align_crop_position) * 2 ** (self.num_scales - 1)
            inputs['crop_info'] = crop_info

            inputs["shuffle_color_aug", 0, 0] = self.layer_crop_shuffle(inputs["color_aug", 0, 0], crop_info)
            shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
            shuffle_outputs = self.net_depth_decoder(shuffle_features)

            for scale in self.scales:
                outputs[("shuffle_disp", scale)] = self.layer_crop_shuffle(shuffle_outputs[("disp", scale)],
                                                                           crop_info // 2 ** scale)

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def layer_shuffle(self, input_raw, direction=True):
        # input raw: (b, 3, h, w)
        if direction:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_up, chunk_0_botton = torch.chunk(chunk_0, 2, 2)
            chunk_1_up, chunk_1_botton = torch.chunk(chunk_1, 2, 2)
            chunk_0_up_1_botton = torch.cat([chunk_0_up, chunk_1_botton], 2)
            chunk_1_up_0_botton = torch.cat([chunk_1_up, chunk_0_botton], 2)
            shuffle_input = torch.cat([chunk_0_up_1_botton, chunk_1_up_0_botton], 0)
        else:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_left, chunk_0_right = torch.chunk(chunk_0, 2, 3)
            chunk_1_left, chunk_1_right = torch.chunk(chunk_1, 2, 3)
            chunk_0_left_1_right = torch.cat([chunk_0_left, chunk_1_right], 3)
            chunk_1_left_0_right = torch.cat([chunk_1_left, chunk_0_right], 3)
            shuffle_input = torch.cat([chunk_0_left_1_right, chunk_1_left_0_right], 0)
        return shuffle_input

    def layer_crop_shuffle(self, input_raw, crop_info):
        chunk_0, chunk_1 = torch.chunk(input_raw, 2)
        x1, y1, x1p, y1p, patch_width, patch_height, w, h = crop_info  # [232 136  64   8  56  80 320 256]

        split_xp = [x1p, patch_width, w - (x1p + patch_width)]
        split_yp = [y1p, patch_height, h - (y1p + patch_height)]

        chunk_0_left_middle_right = torch.split(chunk_0, split_xp, dim=3)
        chunk_0_middle_up_middle_center_middle_right = torch.split(chunk_0_left_middle_right[1], split_yp, dim=2)

        split_x = [x1, patch_width, w - (x1 + patch_width)]
        split_y = [y1, patch_height, h - (y1 + patch_height)]

        chunk_1_left_middle_right = torch.split(chunk_1, split_x, dim=3)
        chunk_1_middle_up_middle_center_middle_right = torch.split(chunk_1_left_middle_right[1], split_y, dim=2)

        shuffle_0_middle = torch.cat(
            [chunk_0_middle_up_middle_center_middle_right[0], chunk_1_middle_up_middle_center_middle_right[1],
             chunk_0_middle_up_middle_center_middle_right[2]], 2)
        shuffle_0 = torch.cat([chunk_0_left_middle_right[0], shuffle_0_middle, chunk_0_left_middle_right[2]], 3)

        shuffle_1_middle = torch.cat(
            [chunk_1_middle_up_middle_center_middle_right[0], chunk_0_middle_up_middle_center_middle_right[1],
             chunk_1_middle_up_middle_center_middle_right[2]], 2)
        shuffle_1 = torch.cat([chunk_1_left_middle_right[0], shuffle_1_middle, chunk_1_left_middle_right[2]], 3)

        shuffle_input = torch.cat([shuffle_0, shuffle_1], 0)
        return shuffle_input

    def get_crop_info(self, h, w, min_patch_ratio=0.6, max_path_ratio=0.8, align_crop_position=False):
        min_width = round(min_patch_ratio * w)
        max_width = round(max_path_ratio * w)

        min_height = round(min_patch_ratio * h)
        max_height = round(max_path_ratio * h)

        patch_width = np.random.randint(min_width, max_width + 1)
        patch_height = np.random.randint(min_height, max_height + 1)

        x1 = np.random.randint(0, w - patch_width)
        y1 = np.random.randint(0, h - patch_height)
        # x2 = x1 + patch_width
        # y2 = y1 + patch_height

        if align_crop_position:
            x1p = x1
            y1p = y1
        else:
            x1p = np.random.randint(0, w - patch_width)
            y1p = np.random.randint(0, h - patch_height)
        # x2p = x1p + patch_width
        # y2p = y1p + patch_height

        return np.array([x1, y1, x1p, y1p, patch_width, patch_height, w, h], dtype=int)

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.net_pose_encoder(torch.cat(pose_inputs, 1))]

                axisangle, translation, intermediate_feature = self.net_pose_decoder(pose_inputs)

                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                cam_K = self.net_depth_intrinsics(intermediate_feature, self.width, self.height)
                inv_K = torch.inverse(cam_K)
                outputs[('K', 0)] = cam_K
                outputs[('inv_K', 0)] = inv_K
        else:
            raise NotImplementedError

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))
                cam_points = _backproject_depth(
                    depth, outputs[('inv_K', 0)])
                _project_3d = getattr(self, "project_3d_{}".format(source_scale))

                if self.geometry_loss:
                    pix_coords, computed_depth = _project_3d(
                        cam_points, inputs[("K", source_scale)], T, compute_depth=True)
                    outputs[('computed_depth', frame_id, scale)] = computed_depth
                    if self.geometry_loss_disp_mode:
                        computed_disp = depth_to_disp(computed_depth, self.min_depth, self.max_depth)
                        outputs[('computed_disp', frame_id, scale)] = computed_disp
                else:
                    pix_coords = _project_3d(
                        cam_points, outputs[('K', 0)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if self.geometry_loss:
                    outputs[("sampled_depth", frame_id, scale)] = F.grid_sample(
                        outputs[("depth", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)
                    if self.geometry_loss_disp_mode:
                        outputs[("sampled_disp", frame_id, scale)] = F.grid_sample(
                            outputs[("disp", frame_id, source_scale)],
                            outputs[("sample", frame_id, scale)],
                            padding_mode="border", align_corners=True)

                outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

        for scale in [0]:
            source_scale = 0
            disp = outputs[("disp_reshuffle", scale)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            for _, frame_id in enumerate(self.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))

                cam_points = _backproject_depth(
                    depth, outputs[('inv_K', 0)])
                _project_3d = getattr(self, "project_3d_{}".format(source_scale))
                pix_coords = _project_3d(
                    cam_points, outputs[('K', 0)], T)
                outputs[("color_reshuffle", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)].clone(),
                    pix_coords,
                    padding_mode="border", align_corners=True)

            disp = outputs[("disp_local", scale)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            for _, frame_id in enumerate(self.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                Rt_Rc = torch.zeros_like(T).to(self.device)
                gx0 = (inputs[("grid_local")][:, 0, 0, -1] + inputs[("grid_local")][:, 0, 0, 0]) / 2.
                gy0 = (inputs[("grid_local")][:, 1, -1, 0] + inputs[("grid_local")][:, 1, 0, 0]) / 2.
                f = (inputs[("grid_local")][:, 0, 0, -1] - inputs[("grid_local")][:, 0, 0, 0]) / 2.
                fx = inputs[("K", 0)][0, 0, 0] / self.width
                fy = inputs[("K", 0)][0, 1, 1] / self.height
                Rc_v = torch.stack([-gx0 / (2 * fx), -gy0 / (2 * fy), f], dim=1)
                Rc = torch.eye(3).to(self.device)
                Rc = Rc[None, :, :].repeat(Rc_v.shape[0], 1, 1)
                Rc[:, :, 2] = Rc_v
                # outputs[("Rc", f_i)] = Rc
                Rt_Rc[:, :3, :3] = torch.matmul(Rc, torch.matmul(T[:, :3, :3], torch.inverse(Rc)))
                Rt_Rc[:, :3, 3:4] = torch.matmul(Rc, T[:, :3, 3:4])
                T = Rt_Rc

                _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))

                cam_points = _backproject_depth(
                    depth, outputs[('inv_K', 0)])
                _project_3d = getattr(self, "project_3d_{}".format(source_scale))

                pix_coords = _project_3d(
                    cam_points, outputs[('K', 0)], T)
                outputs[("color_local", frame_id, scale)] = F.grid_sample(
                    inputs[("color_local", frame_id, source_scale)],
                    pix_coords, padding_mode="border", align_corners=True)

    def compute_losses_local(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in [0]:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp_local", scale)]
            color = inputs[("color_local", 0, scale)]
            target = inputs[("color_local", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color_local", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color_local", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss

        return total_loss

    def compute_losses_reshuffle(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in [0]:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp_reshuffle", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color_reshuffle", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss

        return total_loss

    def compute_losses_ori(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
        total_loss /= 4

        return total_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0
        total_loss1 = 0
        """
        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/photometric_{}".format(scale)] = loss
        """
        total_loss1 += self.compute_losses_ori(inputs, outputs)
        losses["loss/photometric_ori{}"] = self.compute_losses_ori(inputs, outputs)

        total_loss1 += self.compute_losses_local(inputs, outputs) * 0.1
        losses["loss/photometric_local{}"] = self.compute_losses_local(inputs, outputs)
        """

        total_loss1 += self.compute_losses_reshuffle(inputs, outputs)
        losses["loss/photometric_reshuffle{}"] = self.compute_losses_reshuffle(inputs, outputs)
        """

        # total_loss1 /= 2
        """




        loss_dc = torch.tensor(0.0).to(self.device)
        loss_dc_local = 0
        for i in range(1):
            disp = outputs[("disp", i)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            loss_dc_i = 0
            for b in range(self.batch_size):
                disp_local = outputs[("disp_local", i)][b].clone()
                x0 = round(self.width * (inputs[("grid_local")][b, 0, 0, 0].item() - (-1)) / 2.)
                y0 = round(self.height * (inputs[("grid_local")][b, 1, 0, 0].item() - (-1)) / 2.)
                w = round(self.width / inputs[("ratio_local")][b, 0].item())
                h = round(self.height / inputs[("ratio_local")][b, 0].item())
                disp_local = F.interpolate(disp_local.unsqueeze(0), [h, w], mode="bilinear", align_corners=False)
                _, depth_local = disp_to_depth(disp_local, self.min_depth, self.max_depth)
                depth_local *= inputs[("ratio_local")][b, 0]
                _, depth_from_ori = disp_to_depth(disp[b, :, y0:y0 + h, x0:x0 + w].clone().unsqueeze(0), self.min_depth,
                                                  self.max_depth)

                loss_dc_i += self.compute_SI_log_depth_loss(depth_local, depth_from_ori)
            loss_dc_i /= self.batch_size
            loss_dc_local += loss_dc_i
        loss_dc_local /= 1
        losses["loss_dc_local"] = loss_dc_local
        loss_dc += loss_dc_local



        loss_dc_reshuffle = 0
        for i in range(1):
            _, depth = disp_to_depth(outputs[("disp", i)].clone(), self.min_depth, self.max_depth)
            disp_restore = outputs[("disp_reshuffle", i)]
            _, depth_restore = disp_to_depth(disp_restore, self.min_depth, self.max_depth)
            loss_dc_reshuffle += self.compute_SI_log_depth_loss(depth_restore, depth)
        loss_dc_reshuffle /= 1
        losses["loss_dc_reshuffle"] = loss_dc_reshuffle
        loss_dc += loss_dc_reshuffle



        losses["loss_dc"] = loss_dc


        total_loss1 = total_loss1 + 0.01 * loss_dc
        """

        if not self.disable_matcher:
            matcher_loss = 0
            for frame_id in self.frame_ids[1:]:
                # correspondences = self.matcher(inputs[('color', 0, source_scale)], inputs[('color', frame_id, source_scale)])
                correspondences = inputs[('correspondences', 0, frame_id)]
                for scale in self.scales:
                    matcher_loss += compute_matcher_errors_from_correspondences(correspondences,
                                                                                outputs[("sample", frame_id, scale)],
                                                                                self.width, self.height,
                                                                                self.batch_size, self.device,
                                                                                confidence=self.confidence,
                                                                                delta=self.matcher_loss_delta)
            matcher_loss *= self.matcher_loss_alpha
            total_loss += matcher_loss
            losses["loss/matcher"] = matcher_loss

        if self.ratio_consistency and outputs["do_shuffle"]:
            ratio_consistency_loss = 0
            for scale in self.scales:
                # ratio_consistency_loss += torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)]).mean()
                ratio_consistency_loss += self.compute_batch_image_shuffle_loss(outputs[("shuffle_disp", scale)],
                                                                                outputs[("disp", scale)],
                                                                                norm=self.ratio_consistency_normalization)
            total_loss += ratio_consistency_loss
            losses["loss/ratio_consistency"] = ratio_consistency_loss
        elif self.ratio_consistency_crop and outputs["do_shuffle"]:
            ratio_consistency_crop_loss = 0
            for scale in self.scales:
                ratio_consistency_crop_loss_tmp = torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)])
                if self.ratio_consistency_normalization:
                    ratio_consistency_crop_loss_tmp /= (outputs[("shuffle_disp", scale)] + outputs[("disp", scale)])
                if self.ratio_consistency_scales_normalization:
                    ratio_consistency_crop_loss_tmp /= (2 ** scale)
                ratio_consistency_crop_loss += self.compute_random_batch_image_shuffle_loss(
                    ratio_consistency_crop_loss_tmp, inputs["crop_info"] // 2 ** scale).mean()
            ratio_consistency_crop_loss *= self.weight_ratio_consistency_crop
            total_loss += ratio_consistency_crop_loss
            losses["loss/ratio_consistency_crop"] = ratio_consistency_crop_loss

        if self.geometry_loss:
            geometry_loss = 0
            for scale in self.scales:
                for frame_id in self.frame_ids[1:]:
                    if self.geometry_loss_disp_mode:
                        geometry_loss += torch.abs(outputs[("computed_disp", frame_id, scale)] - outputs[
                            ("sampled_disp", frame_id, scale)]).mean()
                    else:
                        geometry_loss += (torch.abs(outputs[("computed_depth", frame_id, scale)] - outputs[
                            ("sampled_depth", frame_id, scale)]) / (
                                                  outputs[("computed_depth", frame_id, scale)] + outputs[
                                              ("sampled_depth", frame_id, scale)])).mean() / (2 ** scale)
            geometry_loss *= 0.1
            total_loss += geometry_loss
            losses["loss/geometry_loss"] = geometry_loss

        total_loss /= len(self.scales)
        losses["loss"] = total_loss + total_loss1
        return losses

    def compute_random_batch_image_shuffle_loss(self, l1, crop_info):
        b, _, h, w = l1.shape
        x1, y1, x1p, y1p, patch_width, patch_height, _, _ = crop_info
        mask = torch.ones_like(l1)
        min_x1p = max(x1p - 1, 0)
        max_x1p = min(x1p + patch_width + 1, w)
        min_y1p = max(y1p - 1, 0)
        max_y1p = min(y1p + patch_height + 1, h)
        mask[:b // 2, :, min_y1p:min_y1p + 2, min_x1p:max_x1p] = 0  # top
        mask[:b // 2, :, max_y1p - 2:max_y1p, min_x1p:max_x1p] = 0  # botton
        mask[:b // 2, :, min_y1p:max_y1p, min_x1p:min_x1p + 2] = 0  # left
        mask[:b // 2, :, min_y1p:max_y1p, max_x1p - 2:max_x1p] = 0  # right

        min_x1 = max(x1 - 1, 0)
        max_x1 = min(x1 + patch_width + 1, w)
        min_y1 = max(y1 - 1, 0)
        max_y1 = min(y1 + patch_height + 1, h)
        mask[b // 2:, :, min_y1:min_y1 + 2, min_x1:max_x1] = 0  # top
        mask[b // 2:, :, max_y1 - 2:max_y1, min_x1:max_x1] = 0  # botton
        mask[b // 2:, :, min_y1:max_y1, min_x1:min_x1 + 2] = 0  # left
        mask[b // 2:, :, min_y1:max_y1, max_x1 - 2:max_x1] = 0  # right

        return l1 * mask

    def compute_batch_image_shuffle_loss(self, pred, target, norm=False):
        mask = torch.ones_like(pred)
        b, _, h, w = pred.shape
        mask[:, :, :, w // 2 - 1:w // 2 + 1] = 0
        mask[:, :, h // 2 - 1:h // 2 + 1, :] = 0
        if norm:
            return ((torch.abs(pred - target) / (pred + target)) * mask).mean()
        else:
            return torch.abs((pred - target) * mask).mean()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_SI_log_depth_loss(self, pred, target, mask=None, lamda=0.5):
        # B*1*H*W  ->  B*H*W
        if mask is None:
            mask = torch.ones_like(pred).to(self.device)

        mask = mask[:, 0]
        log_pred = torch.log(pred[:, 0] + 1e-8) * mask
        log_tgt = torch.log(target[:, 0] + 1e-8) * mask

        log_diff = log_pred - log_tgt
        valid_num = mask.sum(1).sum(1) + 1e-8
        log_diff_squre_sum = (log_diff ** 2).sum(1).sum(1)
        log_diff_sum_squre = (log_diff.sum(1).sum(1)) ** 2
        loss = log_diff_squre_sum / valid_num - lamda * log_diff_sum_squre / (valid_num ** 2)

        return loss.mean()

    @torch.no_grad()
    def compute_depth_losses(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_losses = {}
        depth_gt = inputs["depth_gt"]
        b, c, gt_h, gt_w = depth_gt.shape
        mask = depth_gt > 0

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = F.interpolate(
            depth_pred, [gt_h, gt_w], mode="bilinear", align_corners=False)
        depth_pred = depth_pred.detach()

        depth_gt_flatten = depth_gt.view(self.batch_size, -1)
        depth_pred_flatten = depth_pred.view(self.batch_size, -1)
        mask_flatten = mask.view(self.batch_size, -1)

        med_gt, _ = torch.masked_fill(depth_gt_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)
        med_pred, _ = torch.masked_fill(depth_pred_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)

        ratios = med_gt / med_pred
        avg = torch.mean(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios / med)

        depth_losses['ratio/mean'] = np.array(avg.cpu())
        depth_losses['ratio/med'] = np.array(med.cpu())
        depth_losses['ratio/std'] = np.array(std.cpu())

        depth_pred *= ratios[..., None, None]

        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]

        depth_pred = torch.clamp(depth_pred, min=self.min_gt_depth, max=self.max_gt_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            depth_losses[metric] = np.array(depth_errors[i].cpu())

        return depth_losses
```

### 文件: `model_sc.py`

```py
from .networks import *
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
import os
from .networks.layers import *
import random

EPS = 1e-7


class EstimateDepthSC():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.height = cfgs.get('height', 256)
        self.width = cfgs.get('width', 320)
        self.batch_size = cfgs.get('batch_size', 64)
        
        # checking height and width are multiples of 32
        assert self.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.width % 32 == 0, "'width' must be a multiple of 32"
        
        self.device = cfgs.get('device', 'cpu')
        self.scales = cfgs.get('scales', [0,1,2,3])
        self.num_scales = len(self.scales)
        self.frame_ids = cfgs.get('frame_ids', [0,-1,1])
        self.num_pose_frames = 2
        self.disable_automasking = cfgs.get('disable_automasking', False)

        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

        # depth
        self.min_depth = cfgs.get('min_depth', '0.1')
        self.max_depth = cfgs.get('max_depth', '100.0')
        self.min_gt_depth = cfgs.get('min_gt_depth', '0.001')
        self.max_gt_depth = cfgs.get('max_gt_depth', '1.')
        
        self.model_str = cfgs.get('model_str', 'monodepth2')
        self.num_layers = cfgs.get('num_layers', 18)
        self.weights_init = cfgs.get('weights_init', "pretrained")
        if self.model_str == "monodepth2":
            self.net_depth_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained")
            self.net_depth_decoder = DepthDecoder(self.net_depth_encoder.num_ch_enc, self.scales,)
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        elif self.model_str in ["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"]:
            self.drop_path = cfgs.get('drop_path', 0.2)
            self.net_depth_encoder = LiteMono(model=self.model_str, drop_path_rate=self.drop_path, width=self.width, height=self.height)
            self.net_depth_decoder = DepthDecoderV2(self.net_depth_encoder.num_ch_enc, self.scales)
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoderV2(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        elif self.model_str == 'monovit':
            self.net_depth_encoder = mpvit_small()
            self.net_depth_decoder = MonovitDecoder()
            # pose
            self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        else:
            raise NotImplementedError


        # optim
        self.start_epoch = cfgs.get('start_epoch', 0)
        self.num_epochs = cfgs.get('num_epochs', 20)
        self.lr = cfgs.get('lr', [0.0001, 5e-6, 36, 0.0001, 1e-5, 36])
        self.weight_decay = cfgs.get('weight_decay', 0.02)
        self.disparity_smoothness = cfgs.get('disparity_smoothness', 0.001)
        
        # matcher loss
        self.disable_matcher = cfgs.get('disable_matcher', False)
        self.confidence = cfgs.get('confidence', 0.9)
        self.matcher_loss_alpha = cfgs.get('matcher_loss_alpha', 0.2)
        self.half_epoch_matcher = cfgs.get('half_epoch_matcher', False)
        self.matcher_loss_delta = cfgs.get('matcher_loss_delta', 0)
        
        self.network_names = [k for k in vars(self) if k.startswith('net')]
        # optimizer
        if self.model_str == 'monovit' or 'lite-mono':
            self.make_optimizer = lambda optim_dict: torch.optim.AdamW(
                optim_dict["parameters"], lr=optim_dict["lr"], weight_decay=self.weight_decay)
        elif self.model_str == "monodepth2":
            self.make_optimizer = lambda optim_dict: torch.optim.Adam(
                optim_dict["parameters"], lr=optim_dict["lr"])
        
        # ratio consistency
        self.ratio_consistency = cfgs.get('ratio_consistency', False)
        self.ratio_consistency_crop = cfgs.get('ratio_consistency_crop', False)
        self.ratio_consistency_normalization = cfgs.get('ratio_consistency_normalization', False)
        self.ratio_consistency_scales_normalization = cfgs.get('ratio_consistency_scales_normalization', False)
        self.weight_ratio_consistency_crop = cfgs.get('weight_ratio_consistency_crop', 1.0)
        self.align_crop_position = cfgs.get('align_crop_position', False)
        
        # geometry loss
        self.geometry_loss = cfgs.get('geometry_loss', False)
        self.geometry_loss_disp_mode = cfgs.get('geometry_loss_disp_mode', False)

        # load
        self.load_weights_folder = cfgs.get('load_weights_folder', None)
        self.mypretrain = cfgs.get('mypretrain', None)
        self.not_load_nets = cfgs.get('not_load_nets', ())
        self.not_load_optimizer = cfgs.get('not_load_optimizer', ())
        self.models_to_load = cfgs.get('models_to_load', [])
        
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        
        # no grad layers
        self.ssim = SSIM()
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)
            setattr(self, "backproject_depth_{}".format(scale), BackprojectDepth(self.batch_size, h, w))
            setattr(self, "project_3d_{}".format(scale), Project3D(self.batch_size, h, w))
            
        self.other_param_names = ['ssim']
        for scale in self.scales:
            self.other_param_names += ["backproject_depth_{}".format(scale), "project_3d_{}".format(scale)]

    def init_optimizers(self):
        # optim
        self.optimizer_names = []
        self.parameters_depth = []
        self.parameters_pose = []
        for net_name in self.network_names:
            if not any([p.requires_grad for p in getattr(self, net_name).parameters()]):
                continue
            if net_name.startswith('net_depth'):
                self.parameters_depth += list(getattr(self, net_name).parameters())
            elif net_name.startswith('net_pose'):
                self.parameters_pose += list(getattr(self, net_name).parameters())
        self.optimizer_depth = self.make_optimizer({"parameters": self.parameters_depth, "lr": self.lr[0]})
        self.optimizer_pose = self.make_optimizer({"parameters": self.parameters_pose, "lr": self.lr[3]})
        self.optimizer_names += ["optimizer_depth", "optimizer_pose"]
        
        # scheduler
        self.scheduler_names = []
        if self.model_str == 'lite-mono':
            self.scheduler_depth_lr = ChainedScheduler(
                self.optimizer_depth,
                T_0=int(self.lr[2]), T_mul=1, eta_min=self.lr[1], last_epoch=self.start_epoch-1,
                max_lr=self.lr[0], warmup_steps=0, gamma=0.9)
            self.scheduler_pose_lr = ChainedScheduler(
                self.optimizer_pose,
                T_0=int(self.lr[5]), T_mul=1, eta_min=self.lr[4], last_epoch=self.start_epoch-1,
                max_lr=self.lr[3], warmup_steps=0, gamma=0.9)
        elif self.model_str == 'monovit':
            self.scheduler_depth_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_depth, 0.9)
            self.scheduler_pose_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pose, 0.9)
        elif self.model_str == 'monodepth2':
            self.scheduler_depth_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_depth, self.num_epochs - 5, 0.1)
            self.scheduler_pose_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_pose, self.num_epochs - 5, 0.1)

        self.scheduler_names += ["scheduler_depth_lr", "scheduler_pose_lr"]

    def load_model(self):
        """Load model(s) from disk
        """
        self.load_weights_folder = os.path.expanduser(self.load_weights_folder)

        assert os.path.isdir(self.load_weights_folder), \
            "Cannot find folder {}".format(self.load_weights_folder)
        print("loading model from folder {}".format(self.load_weights_folder))

        for n in self.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.load_weights_folder, "{}.pth".format(n))

            model_dict = getattr(self, n).state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            getattr(self, n).load_state_dict(model_dict)

        # loading optimizer state
        optimizer_depth_load_path = os.path.join(self.load_weights_folder, "optimizer_depth.pth")
        optimizer_pose_load_path = os.path.join(self.load_weights_folder, "optimizer_pose.pth")
        if os.path.isfile(optimizer_depth_load_path) and os.path.isfile(optimizer_pose_load_path):
            print("Loading optimizer weights")
            self.optimizer_depth.load_state_dict(torch.load(optimizer_depth_load_path))
            self.optimizer_pose.load_state_dict(torch.load(optimizer_pose_load_path))
        else:
            print("Cannot find optimizer weights so Adam is randomly initialized")

    def load_pretrain(self):
        # only designed for lite-mono
        self.mypretrain = os.path.expanduser(self.mypretrain)
        path = self.mypretrain
        model_dict = self.net_depth_encoder.state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.net_depth_encoder.load_state_dict(model_dict)
        print('mypretrain loaded.')
    
    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names and k not in self.not_load_nets:
                print("Loading ", k)
                model_dict = getattr(self, k).state_dict()
                getattr(self, k).load_state_dict({k: v for k, v in cp[k].items() if k in model_dict})

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names and k not in self.not_load_optimizer:
                print("Loading ", k)
                getattr(self, k).load_state_dict(cp[k])
                
    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
            
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def backward(self, losses):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        losses["loss"].backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def forward(self, inputs):
        """Feedforward once."""
        for key, ipt in inputs.items():
            if "correspondences" in key:
                inputs[key] = ipt
            else:
                inputs[key] = ipt.to(self.device)
        
        # we only feed the image with frame_id 0 through the depth encoder
        features = self.net_depth_encoder(inputs["color_aug", 0, 0])
        outputs = self.net_depth_decoder(features)
        outputs.update(self.predict_poses(inputs))
        

        
        # shuffle
        outputs["do_shuffle"] = random.random() > 0.5
        if self.ratio_consistency and outputs["do_shuffle"]:
            direction = random.random() > 0.5
            inputs["shuffle_color_aug", 0, 0] = self.layer_shuffle(inputs["color_aug", 0, 0], direction)
            shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
            shuffle_outputs = self.net_depth_decoder(shuffle_features)
            for scale in self.scales:
                outputs[("shuffle_disp", scale)] = self.layer_shuffle(shuffle_outputs[("disp", scale)], direction)
        elif self.ratio_consistency_crop and outputs["do_shuffle"]:
            b, _, h, w = inputs["color_aug", 0, 0].shape
            crop_info = self.get_crop_info(h // 2**(self.num_scales-1), w // 2**(self.num_scales-1), align_crop_position=self.align_crop_position) * 2**(self.num_scales-1)
            inputs['crop_info'] = crop_info

            inputs["shuffle_color_aug", 0, 0] = self.layer_crop_shuffle(inputs["color_aug", 0, 0], crop_info)
            shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
            shuffle_outputs = self.net_depth_decoder(shuffle_features)
            for scale in self.scales:
                outputs[("shuffle_disp", scale)] = self.layer_crop_shuffle(shuffle_outputs[("disp", scale)], crop_info // 2**scale)
        
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses
    
    def layer_shuffle(self, input_raw, direction=True):
        # input raw: (b, 3, h, w)
        if direction:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_up, chunk_0_botton = torch.chunk(chunk_0, 2, 2)
            chunk_1_up, chunk_1_botton = torch.chunk(chunk_1, 2, 2)
            chunk_0_up_1_botton = torch.cat([chunk_0_up, chunk_1_botton], 2)
            chunk_1_up_0_botton = torch.cat([chunk_1_up, chunk_0_botton], 2)
            shuffle_input = torch.cat([chunk_0_up_1_botton, chunk_1_up_0_botton], 0)
        else:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_left, chunk_0_right = torch.chunk(chunk_0, 2, 3)
            chunk_1_left, chunk_1_right = torch.chunk(chunk_1, 2, 3)
            chunk_0_left_1_right = torch.cat([chunk_0_left, chunk_1_right], 3)
            chunk_1_left_0_right = torch.cat([chunk_1_left, chunk_0_right], 3)
            shuffle_input = torch.cat([chunk_0_left_1_right, chunk_1_left_0_right], 0)
        return shuffle_input
    
    def layer_crop_shuffle(self, input_raw, crop_info):
        chunk_0, chunk_1 = torch.chunk(input_raw, 2)
        x1, y1, x1p, y1p, patch_width, patch_height, w, h = crop_info # [232 136  64   8  56  80 320 256] 
        
        split_xp = [x1p, patch_width, w - (x1p + patch_width)]
        split_yp = [y1p, patch_height, h - (y1p + patch_height)]
        
        chunk_0_left_middle_right = torch.split(chunk_0, split_xp, dim=3)
        chunk_0_middle_up_middle_center_middle_right = torch.split(chunk_0_left_middle_right[1], split_yp, dim=2)
        
        split_x = [x1, patch_width, w - (x1 + patch_width)]
        split_y = [y1, patch_height, h - (y1 + patch_height)]
        
        chunk_1_left_middle_right = torch.split(chunk_1, split_x, dim=3)
        chunk_1_middle_up_middle_center_middle_right = torch.split(chunk_1_left_middle_right[1], split_y, dim=2)
        
        shuffle_0_middle = torch.cat([chunk_0_middle_up_middle_center_middle_right[0], chunk_1_middle_up_middle_center_middle_right[1], chunk_0_middle_up_middle_center_middle_right[2]], 2)
        shuffle_0 = torch.cat([chunk_0_left_middle_right[0], shuffle_0_middle, chunk_0_left_middle_right[2]], 3)
        
        shuffle_1_middle = torch.cat([chunk_1_middle_up_middle_center_middle_right[0], chunk_0_middle_up_middle_center_middle_right[1], chunk_1_middle_up_middle_center_middle_right[2]], 2)
        shuffle_1 = torch.cat([chunk_1_left_middle_right[0], shuffle_1_middle, chunk_1_left_middle_right[2]], 3)
        
        shuffle_input = torch.cat([shuffle_0, shuffle_1], 0)
        return shuffle_input

    def get_crop_info(self, h, w, min_patch_ratio=0.6, max_path_ratio=0.8, align_crop_position=False):
        min_width = round(min_patch_ratio * w)
        max_width = round(max_path_ratio * w)
        
        min_height = round(min_patch_ratio * h)
        max_height = round(max_path_ratio * h)
        
        patch_width = np.random.randint(min_width, max_width+1)
        patch_height = np.random.randint(min_height, max_height+1)
        
        x1 = np.random.randint(0, w - patch_width)
        y1 = np.random.randint(0, h - patch_height)
        # x2 = x1 + patch_width
        # y2 = y1 + patch_height
        
        if align_crop_position:
            x1p = x1
            y1p = y1
        else:
            x1p = np.random.randint(0, w - patch_width)
            y1p = np.random.randint(0, h - patch_height)
        # x2p = x1p + patch_width
        # y2p = y1p + patch_height

        return np.array([x1, y1, x1p, y1p, patch_width, patch_height, w, h], dtype=int)

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.net_pose_encoder(torch.cat(pose_inputs, 1))]
                axisangle, translation = self.net_pose_decoder(pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            raise NotImplementedError

        return outputs
    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))
                cam_points = _backproject_depth(
                    depth, inputs[("inv_K", source_scale)])
                _project_3d = getattr(self, "project_3d_{}".format(source_scale))
                
                pix_coords = _project_3d(
                        cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="zeros", align_corners=True)

                outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]
    
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            valid_masks = []
            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                valid_masks.append((pred.abs().mean(1, True) > 1e-3).float())

            reprojection_losses = torch.cat(reprojection_losses, 1)
            valid_masks = torch.cat(valid_masks + valid_masks, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    # if camera does not move, pred and target are the same, so that loss=0
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

            reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            # [not move, corretlymatch]
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            to_optimise, idxs = torch.min(combined, dim=1, keepdim=True)
            valid_mask = torch.gather(valid_masks, 1, idxs)

            # true means corretly match, false means not move
            outputs["identity_selection/{}".format(scale)] = torch.squeeze(
                idxs > identity_reprojection_loss.shape[1] - 1, dim=1).float()

            loss += self.mean_on_mask(to_optimise, valid_mask)
        
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/photometric_{}".format(scale)] = loss

        if not self.disable_matcher:
            matcher_loss = 0
            for frame_id in self.frame_ids[1:]:
                # correspondences = self.matcher(inputs[('color', 0, source_scale)], inputs[('color', frame_id, source_scale)])
                correspondences = inputs[('correspondences', 0, frame_id)]
                for scale in self.scales:
                    matcher_loss += compute_matcher_errors_from_correspondences(correspondences, outputs[("sample", frame_id, scale)],
                                                                                self.width, self.height, self.batch_size, self.device,
                                                                                confidence=self.confidence, delta=self.matcher_loss_delta)
            matcher_loss *= self.matcher_loss_alpha
            total_loss += matcher_loss
            losses["loss/matcher"] = matcher_loss
            
        if self.ratio_consistency and outputs["do_shuffle"]:
            ratio_consistency_loss = 0
            for scale in self.scales:
                # ratio_consistency_loss += torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)]).mean()
                ratio_consistency_loss += self.compute_batch_image_shuffle_loss(outputs[("shuffle_disp", scale)], outputs[("disp", scale)], norm=self.ratio_consistency_normalization)
            total_loss += ratio_consistency_loss
            losses["loss/ratio_consistency"] = ratio_consistency_loss
        elif self.ratio_consistency_crop and outputs["do_shuffle"]:
            ratio_consistency_crop_loss = 0
            for scale in self.scales:
                ratio_consistency_crop_loss_tmp = torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)])
                if self.ratio_consistency_normalization:
                    ratio_consistency_crop_loss_tmp /= (outputs[("shuffle_disp", scale)] + outputs[("disp", scale)])
                if self.ratio_consistency_scales_normalization:
                    ratio_consistency_crop_loss_tmp /= (2 ** scale)
                ratio_consistency_crop_loss += self.compute_random_batch_image_shuffle_loss(ratio_consistency_crop_loss_tmp, inputs["crop_info"] // 2**scale).mean()
            ratio_consistency_crop_loss *= self.weight_ratio_consistency_crop
            total_loss += ratio_consistency_crop_loss
            losses["loss/ratio_consistency_crop"] = ratio_consistency_crop_loss

        total_loss /= len(self.scales)
        losses["loss"] = total_loss
        return losses

    def mean_on_mask(self, diff, valid_mask):
        mask = valid_mask.expand_as(diff)
        if mask.sum() > 100:
            mean_value = (diff * mask).sum() / mask.sum()
        else:
            mean_value = torch.tensor(0).float().to(self.device)
        return mean_value

    def compute_random_batch_image_shuffle_loss(self, l1, crop_info):
        b, _, h, w = l1.shape
        x1, y1, x1p, y1p, patch_width, patch_height, _, _ = crop_info
        mask = torch.ones_like(l1)
        min_x1p = max(x1p - 1, 0)
        max_x1p = min(x1p + patch_width + 1, w)
        min_y1p = max(y1p - 1, 0)
        max_y1p = min(y1p + patch_height + 1, h)
        mask[:b//2, :, min_y1p:min_y1p+2, min_x1p:max_x1p] = 0 # top
        mask[:b//2, :, max_y1p-2:max_y1p, min_x1p:max_x1p] = 0 # botton
        mask[:b//2, :, min_y1p:max_y1p, min_x1p:min_x1p+2] = 0 # left
        mask[:b//2, :, min_y1p:max_y1p, max_x1p-2:max_x1p] = 0 # right

        min_x1 = max(x1 - 1, 0)
        max_x1 = min(x1 + patch_width + 1, w)
        min_y1 = max(y1 - 1, 0)
        max_y1 = min(y1 + patch_height + 1, h)
        mask[b//2:, :, min_y1:min_y1+2, min_x1:max_x1] = 0 # top
        mask[b//2:, :, max_y1-2:max_y1, min_x1:max_x1] = 0 # botton
        mask[b//2:, :, min_y1:max_y1, min_x1:min_x1+2] = 0 # left
        mask[b//2:, :, min_y1:max_y1, max_x1-2:max_x1] = 0 # right

        return l1 * mask
    
    def compute_batch_image_shuffle_loss(self, pred, target, norm=False):
        mask = torch.ones_like(pred)
        b, _, h, w = pred.shape
        mask[:, :, :, w//2-1:w//2+1] = 0
        mask[:, :, h//2-1:h//2+1, :] = 0
        if norm:
            return ((torch.abs(pred - target) / (pred + target)) * mask).mean()
        else:
            return torch.abs((pred - target) * mask).mean()
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    @torch.no_grad()
    def compute_depth_losses(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_losses = {}
        depth_gt = inputs["depth_gt"]
        b, c, gt_h, gt_w = depth_gt.shape
        mask = depth_gt > 0
        
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = F.interpolate(
            depth_pred, [gt_h, gt_w], mode="bilinear", align_corners=False)
        depth_pred = depth_pred.detach()

        
        depth_gt_flatten = depth_gt.view(self.batch_size, -1)
        depth_pred_flatten = depth_pred.view(self.batch_size, -1)
        mask_flatten = mask.view(self.batch_size, -1)
        
        med_gt, _ = torch.masked_fill(depth_gt_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)
        med_pred, _ = torch.masked_fill(depth_pred_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)

        ratios = med_gt / med_pred
        avg = torch.mean(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios/med)

        depth_losses['ratio/mean'] = np.array(avg.cpu())
        depth_losses['ratio/med'] = np.array(med.cpu())
        depth_losses['ratio/std'] = np.array(std.cpu())
        
        depth_pred *= ratios[..., None, None]
        
        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]

        depth_pred = torch.clamp(depth_pred, min=self.min_gt_depth, max=self.max_gt_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            depth_losses[metric] = np.array(depth_errors[i].cpu())
            
        return depth_losses

```

### 文件: `model_supervised.py`

```py
from .networks import *
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
import os
from .networks.layers import *
import random

EPS = 1e-7


class EstimateDepthSupervised():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.height = cfgs.get('height', 256)
        self.width = cfgs.get('width', 320)
        self.batch_size = cfgs.get('batch_size', 64)
        
        # checking height and width are multiples of 32
        assert self.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.width % 32 == 0, "'width' must be a multiple of 32"
        
        self.device = cfgs.get('device', 'cpu')
        self.scales = cfgs.get('scales', [0,1,2,3])
        self.num_scales = len(self.scales)
        self.frame_ids = cfgs.get('frame_ids', [0,-1,1])
        self.num_pose_frames = 2
        self.disable_automasking = cfgs.get('disable_automasking', False)

        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

        # depth
        self.min_depth = cfgs.get('min_depth', 0.1)
        self.max_depth = cfgs.get('max_depth', 100.0)
        self.min_gt_depth = cfgs.get('min_gt_depth', 0.001)
        self.max_gt_depth = cfgs.get('max_gt_depth', 1.)
        
        self.model_str = cfgs.get('model_str', 'monodepth2')
        self.scheduler_str = cfgs.get('scheduler_str', None)
        self.num_layers = cfgs.get('num_layers', 18)
        self.weights_init = cfgs.get('weights_init', "pretrained")
        if self.model_str == "monodepth2":
            self.net_depth_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained")
            self.net_depth_decoder = DepthDecoder(self.net_depth_encoder.num_ch_enc, self.scales,)
            # # pose
            # self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            # self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        elif self.model_str in ["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"]:
            self.drop_path = cfgs.get('drop_path', 0.2)
            self.net_depth_encoder = LiteMono(model=self.model_str, drop_path_rate=self.drop_path, width=self.width, height=self.height)
            self.net_depth_decoder = DepthDecoderV2(self.net_depth_encoder.num_ch_enc, self.scales)
            # # pose
            # self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            # self.net_pose_decoder = PoseDecoderV2(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        elif self.model_str == 'monovit':
            self.net_depth_encoder = mpvit_small()
            self.net_depth_decoder = MonovitDecoder()
            # # pose
            # self.net_pose_encoder = ResnetEncoder(self.num_layers, self.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            # self.net_pose_decoder = PoseDecoder(self.net_pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        else:
            raise NotImplementedError


        # optim
        self.start_epoch = cfgs.get('start_epoch', 0)
        self.num_epochs = cfgs.get('num_epochs', 20)
        self.lr = cfgs.get('lr', [0.0001, 5e-6, 36, 0.0001, 1e-5, 36])
        self.weight_decay = cfgs.get('weight_decay', 0.02)
        self.disparity_smoothness = cfgs.get('disparity_smoothness', 0.001)
        
        # # matcher loss
        # self.disable_matcher = cfgs.get('disable_matcher', False)
        # self.confidence = cfgs.get('confidence', 0.9)
        # self.matcher_loss_alpha = cfgs.get('matcher_loss_alpha', 0.2)
        # self.half_epoch_matcher = cfgs.get('half_epoch_matcher', False)
        # self.matcher_loss_delta = cfgs.get('matcher_loss_delta', 0)
        
        self.network_names = [k for k in vars(self) if k.startswith('net')]
        # optimizer
        if self.model_str == 'monovit' or 'lite-mono':
            self.make_optimizer = lambda optim_dict: torch.optim.AdamW(
                optim_dict["parameters"], lr=optim_dict["lr"], weight_decay=self.weight_decay)
        elif self.model_str == "monodepth2":
            self.make_optimizer = lambda optim_dict: torch.optim.Adam(
                optim_dict["parameters"], lr=optim_dict["lr"])
        
        # # ratio consistency
        # self.ratio_consistency = cfgs.get('ratio_consistency', False)
        # self.ratio_consistency_crop = cfgs.get('ratio_consistency_crop', False)
        # self.ratio_consistency_normalization = cfgs.get('ratio_consistency_normalization', False)
        # self.ratio_consistency_scales_normalization = cfgs.get('ratio_consistency_scales_normalization', False)
        # self.weight_ratio_consistency_crop = cfgs.get('weight_ratio_consistency_crop', 1.0)
        # self.align_crop_position = cfgs.get('align_crop_position', False)
        
        # # geometry loss
        # self.geometry_loss = cfgs.get('geometry_loss', False)
        # self.geometry_loss_disp_mode = cfgs.get('geometry_loss_disp_mode', False)

        # load
        self.load_weights_folder = cfgs.get('load_weights_folder', None)
        self.mypretrain = cfgs.get('mypretrain', None)
        self.not_load_nets = cfgs.get('not_load_nets', ())
        self.not_load_optimizer = cfgs.get('not_load_optimizer', ())
        self.models_to_load = cfgs.get('models_to_load', [])
        
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        
        # no grad layers
        self.ssim = SSIM()
        for scale in self.scales:
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)
            setattr(self, "backproject_depth_{}".format(scale), BackprojectDepth(self.batch_size, h, w))
            setattr(self, "project_3d_{}".format(scale), Project3D(self.batch_size, h, w))
            
        self.other_param_names = ['ssim']
        for scale in self.scales:
            self.other_param_names += ["backproject_depth_{}".format(scale), "project_3d_{}".format(scale)]

    def init_optimizers(self):
        # optim
        self.optimizer_names = []
        self.parameters_depth = []
        self.parameters_pose = []
        for net_name in self.network_names:
            if not any([p.requires_grad for p in getattr(self, net_name).parameters()]):
                continue
            if net_name.startswith('net_depth'):
                self.parameters_depth += list(getattr(self, net_name).parameters())
            # elif net_name.startswith('net_pose'):
            #     self.parameters_pose += list(getattr(self, net_name).parameters())
        self.optimizer_depth = self.make_optimizer({"parameters": self.parameters_depth, "lr": self.lr[0]})
        # self.optimizer_pose = self.make_optimizer({"parameters": self.parameters_pose, "lr": self.lr[3]})
        self.optimizer_names += ["optimizer_depth"]
        # self.optimizer_names += ["optimizer_depth", "optimizer_pose"]
        
        # scheduler
        self.scheduler_names = []
        if self.scheduler_str is not None:
            print("scheduler_str mode...")
            if self.scheduler_str == 'cosine':
                self.scheduler_depth_lr = ChainedScheduler(
                    self.optimizer_depth,
                    T_0=int(self.lr[2]), T_mul=1, eta_min=self.lr[1], last_epoch=self.start_epoch-1,
                    max_lr=self.lr[0], warmup_steps=0, gamma=0.9)
                # self.scheduler_pose_lr = ChainedScheduler(
                #     self.optimizer_pose,
                #     T_0=int(self.lr[5]), T_mul=1, eta_min=self.lr[4], last_epoch=self.start_epoch-1,
                #     max_lr=self.lr[3], warmup_steps=0, gamma=0.9)
                print("optimiser cosine")
            elif self.scheduler_str == 'exp':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_depth, 0.9)
                # self.scheduler_pose_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pose, 0.9)
                print("optimiser exp")
            elif self.scheduler_str == 'step':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_depth, self.num_epochs - 5, 0.1)
                # self.scheduler_pose_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_pose, self.num_epochs - 5, 0.1)
                print("optimiser step")
        else:
            if self.model_str == 'lite-mono':
                self.scheduler_depth_lr = ChainedScheduler(
                    self.optimizer_depth,
                    T_0=int(self.lr[2]), T_mul=1, eta_min=self.lr[1], last_epoch=self.start_epoch-1,
                    max_lr=self.lr[0], warmup_steps=0, gamma=0.9)
                # self.scheduler_pose_lr = ChainedScheduler(
                #     self.optimizer_pose,
                #     T_0=int(self.lr[5]), T_mul=1, eta_min=self.lr[4], last_epoch=self.start_epoch-1,
                #     max_lr=self.lr[3], warmup_steps=0, gamma=0.9)
                print("optimiser cosine")
            elif self.model_str == 'monovit':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_depth, 0.9)
                # self.scheduler_pose_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_pose, 0.9)
                print("optimiser exp")
            elif self.model_str == 'monodepth2':
                self.scheduler_depth_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_depth, self.num_epochs - 5, 0.1)
                # self.scheduler_pose_lr = torch.optim.lr_scheduler.StepLR(self.optimizer_pose, self.num_epochs - 5, 0.1)
                print("optimiser step")

        self.scheduler_names += ["scheduler_depth_lr"]
        # self.scheduler_names += ["scheduler_depth_lr", "scheduler_pose_lr"]

    def load_model(self):
        """Load model(s) from disk
        """
        self.load_weights_folder = os.path.expanduser(self.load_weights_folder)

        assert os.path.isdir(self.load_weights_folder), \
            "Cannot find folder {}".format(self.load_weights_folder)
        print("loading model from folder {}".format(self.load_weights_folder))

        for n in self.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.load_weights_folder, "{}.pth".format(n))

            model_dict = getattr(self, n).state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            getattr(self, n).load_state_dict(model_dict)

        # loading optimizer state
        optimizer_depth_load_path = os.path.join(self.load_weights_folder, "optimizer_depth.pth")
        # optimizer_pose_load_path = os.path.join(self.load_weights_folder, "optimizer_pose.pth")
        if os.path.isfile(optimizer_depth_load_path) and os.path.isfile(optimizer_depth_load_path):
            print("Loading optimizer weights")
            self.optimizer_depth.load_state_dict(torch.load(optimizer_depth_load_path))
            # self.optimizer_pose.load_state_dict(torch.load(optimizer_pose_load_path))
        else:
            print("Cannot find optimizer weights so Adam is randomly initialized")

    def load_pretrain(self):
        # only designed for lite-mono
        self.mypretrain = os.path.expanduser(self.mypretrain)
        path = self.mypretrain
        model_dict = self.net_depth_encoder.state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.net_depth_encoder.load_state_dict(model_dict)
        print('mypretrain loaded.')
    
    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names and k not in self.not_load_nets:
                print("Loading ", k)
                model_dict = getattr(self, k).state_dict()
                getattr(self, k).load_state_dict({k: v for k, v in cp[k].items() if k in model_dict})

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names and k not in self.not_load_optimizer:
                print("Loading ", k)
                getattr(self, k).load_state_dict(cp[k])
                
    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
            
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def backward(self, losses):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        losses["loss"].backward()
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).step()

    def forward(self, inputs):
        """Feedforward once."""
        for key, ipt in inputs.items():
            if "correspondences" in key:
                inputs[key] = ipt
            else:
                inputs[key] = ipt.to(self.device)
        
        # we only feed the image with frame_id 0 through the depth encoder
        features = self.net_depth_encoder(inputs["color_aug", 0, 0])
        outputs = self.net_depth_decoder(features)
        # outputs.update(self.predict_poses(inputs))
        
        # # depth in frame [-1, 1] and scales [0, 1, 2, 3]
        # if self.geometry_loss:
        #     for i, frame_id in enumerate(self.frame_ids[1:]):
        #         outputs_disp_scales = self.net_depth_decoder(self.net_depth_encoder(inputs["color_aug", frame_id, 0]))
        #         for scale in self.scales:
        #             disp = outputs_disp_scales[("disp", scale)]
        #             disp = F.interpolate(
        #                 disp, [self.height, self.width], mode="bilinear", align_corners=False)
        #             _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        #             outputs[("depth", frame_id, scale)] = depth
        #             if self.geometry_loss_disp_mode:
        #                 outputs[("disp", frame_id, scale)] = disp
        
        # # shuffle
        # outputs["do_shuffle"] = random.random() > 0.5
        # if self.ratio_consistency and outputs["do_shuffle"]:
        #     direction = random.random() > 0.5
        #     inputs["shuffle_color_aug", 0, 0] = self.layer_shuffle(inputs["color_aug", 0, 0], direction)
        #     shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
        #     shuffle_outputs = self.net_depth_decoder(shuffle_features)
        #     for scale in self.scales:
        #         outputs[("shuffle_disp", scale)] = self.layer_shuffle(shuffle_outputs[("disp", scale)], direction)
        # elif self.ratio_consistency_crop and outputs["do_shuffle"]:
        #     b, _, h, w = inputs["color_aug", 0, 0].shape
        #     crop_info = self.get_crop_info(h // 2**(self.num_scales-1), w // 2**(self.num_scales-1), align_crop_position=self.align_crop_position) * 2**(self.num_scales-1)
        #     inputs['crop_info'] = crop_info

        #     inputs["shuffle_color_aug", 0, 0] = self.layer_crop_shuffle(inputs["color_aug", 0, 0], crop_info)
        #     shuffle_features = self.net_depth_encoder(inputs["shuffle_color_aug", 0, 0])
        #     shuffle_outputs = self.net_depth_decoder(shuffle_features)
        #     for scale in self.scales:
        #         outputs[("shuffle_disp", scale)] = self.layer_crop_shuffle(shuffle_outputs[("disp", scale)], crop_info // 2**scale)
        
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses
    
    def layer_shuffle(self, input_raw, direction=True):
        # input raw: (b, 3, h, w)
        if direction:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_up, chunk_0_botton = torch.chunk(chunk_0, 2, 2)
            chunk_1_up, chunk_1_botton = torch.chunk(chunk_1, 2, 2)
            chunk_0_up_1_botton = torch.cat([chunk_0_up, chunk_1_botton], 2)
            chunk_1_up_0_botton = torch.cat([chunk_1_up, chunk_0_botton], 2)
            shuffle_input = torch.cat([chunk_0_up_1_botton, chunk_1_up_0_botton], 0)
        else:
            chunk_0, chunk_1 = torch.chunk(input_raw, 2)
            chunk_0_left, chunk_0_right = torch.chunk(chunk_0, 2, 3)
            chunk_1_left, chunk_1_right = torch.chunk(chunk_1, 2, 3)
            chunk_0_left_1_right = torch.cat([chunk_0_left, chunk_1_right], 3)
            chunk_1_left_0_right = torch.cat([chunk_1_left, chunk_0_right], 3)
            shuffle_input = torch.cat([chunk_0_left_1_right, chunk_1_left_0_right], 0)
        return shuffle_input
    
    def layer_crop_shuffle(self, input_raw, crop_info):
        chunk_0, chunk_1 = torch.chunk(input_raw, 2)
        x1, y1, x1p, y1p, patch_width, patch_height, w, h = crop_info # [232 136  64   8  56  80 320 256] 
        
        split_xp = [x1p, patch_width, w - (x1p + patch_width)]
        split_yp = [y1p, patch_height, h - (y1p + patch_height)]
        
        chunk_0_left_middle_right = torch.split(chunk_0, split_xp, dim=3)
        chunk_0_middle_up_middle_center_middle_right = torch.split(chunk_0_left_middle_right[1], split_yp, dim=2)
        
        split_x = [x1, patch_width, w - (x1 + patch_width)]
        split_y = [y1, patch_height, h - (y1 + patch_height)]
        
        chunk_1_left_middle_right = torch.split(chunk_1, split_x, dim=3)
        chunk_1_middle_up_middle_center_middle_right = torch.split(chunk_1_left_middle_right[1], split_y, dim=2)
        
        shuffle_0_middle = torch.cat([chunk_0_middle_up_middle_center_middle_right[0], chunk_1_middle_up_middle_center_middle_right[1], chunk_0_middle_up_middle_center_middle_right[2]], 2)
        shuffle_0 = torch.cat([chunk_0_left_middle_right[0], shuffle_0_middle, chunk_0_left_middle_right[2]], 3)
        
        shuffle_1_middle = torch.cat([chunk_1_middle_up_middle_center_middle_right[0], chunk_0_middle_up_middle_center_middle_right[1], chunk_1_middle_up_middle_center_middle_right[2]], 2)
        shuffle_1 = torch.cat([chunk_1_left_middle_right[0], shuffle_1_middle, chunk_1_left_middle_right[2]], 3)
        
        shuffle_input = torch.cat([shuffle_0, shuffle_1], 0)
        return shuffle_input

    def get_crop_info(self, h, w, min_patch_ratio=0.6, max_path_ratio=0.8, align_crop_position=False):
        min_width = round(min_patch_ratio * w)
        max_width = round(max_path_ratio * w)
        
        min_height = round(min_patch_ratio * h)
        max_height = round(max_path_ratio * h)
        
        patch_width = np.random.randint(min_width, max_width+1)
        patch_height = np.random.randint(min_height, max_height+1)
        
        x1 = np.random.randint(0, w - patch_width)
        y1 = np.random.randint(0, h - patch_height)
        # x2 = x1 + patch_width
        # y2 = y1 + patch_height
        
        if align_crop_position:
            x1p = x1
            y1p = y1
        else:
            x1p = np.random.randint(0, w - patch_width)
            y1p = np.random.randint(0, h - patch_height)
        # x2p = x1p + patch_width
        # y2p = y1p + patch_height

        return np.array([x1, y1, x1p, y1p, patch_width, patch_height, w, h], dtype=int)

    # def predict_poses(self, inputs):
    #     """Predict poses between input frames for monocular sequences.
    #     """
    #     outputs = {}
    #     if self.num_pose_frames == 2:
    #         # In this setting, we compute the pose to each source frame via a
    #         # separate forward pass through the pose network.

    #         # select what features the pose network takes as input
    #         pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

    #         for f_i in self.frame_ids[1:]:
    #             # To maintain ordering we always pass frames in temporal order
    #             if f_i < 0:
    #                 pose_inputs = [pose_feats[f_i], pose_feats[0]]
    #             else:
    #                 pose_inputs = [pose_feats[0], pose_feats[f_i]]

    #             pose_inputs = [self.net_pose_encoder(torch.cat(pose_inputs, 1))]
    #             axisangle, translation = self.net_pose_decoder(pose_inputs)
    #             outputs[("axisangle", 0, f_i)] = axisangle
    #             outputs[("translation", 0, f_i)] = translation

    #             # Invert the matrix if the frame id is negative
    #             outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
    #                 axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
    #     else:
    #         raise NotImplementedError

    #     return outputs
    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        outputs[("depth", 0, 0)] = outputs[("disp", 0)]
        # scale = 0
        # disp = outputs[("disp", scale)]        
        # _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        # outputs[("depth", 0, scale)] = depth
        
        # for scale in self.scales:
        #     disp = outputs[("disp", scale)]
        #     disp = F.interpolate(
        #         disp, [self.height, self.width], mode="bilinear", align_corners=False)
        #     source_scale = 0
        #     _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        #     outputs[("depth", 0, scale)] = depth

            # for i, frame_id in enumerate(self.frame_ids[1:]):

            #     T = outputs[("cam_T_cam", 0, frame_id)]
            #     _backproject_depth = getattr(self, "backproject_depth_{}".format(source_scale))
            #     cam_points = _backproject_depth(
            #         depth, inputs[("inv_K", source_scale)])
            #     _project_3d = getattr(self, "project_3d_{}".format(source_scale))
                
            #     if self.geometry_loss:
            #         pix_coords, computed_depth = _project_3d(
            #             cam_points, inputs[("K", source_scale)], T, compute_depth=True)
            #         outputs[('computed_depth', frame_id, scale)] = computed_depth
            #         if self.geometry_loss_disp_mode:
            #             computed_disp = depth_to_disp(computed_depth, self.min_depth, self.max_depth)
            #             outputs[('computed_disp', frame_id, scale)] = computed_disp
            #     else:
            #         pix_coords = _project_3d(
            #             cam_points, inputs[("K", source_scale)], T)

            #     outputs[("sample", frame_id, scale)] = pix_coords

            #     outputs[("color", frame_id, scale)] = F.grid_sample(
            #         inputs[("color", frame_id, source_scale)],
            #         outputs[("sample", frame_id, scale)],
            #         padding_mode="border", align_corners=True)
                
            #     if self.geometry_loss:
            #         outputs[("sampled_depth", frame_id, scale)] = F.grid_sample(
            #             outputs[("depth", frame_id, source_scale)],
            #             outputs[("sample", frame_id, scale)],
            #             padding_mode="border", align_corners=True)
            #         if self.geometry_loss_disp_mode:
            #             outputs[("sampled_disp", frame_id, scale)] = F.grid_sample(
            #             outputs[("disp", frame_id, source_scale)],
            #             outputs[("sample", frame_id, scale)],
            #             padding_mode="border", align_corners=True)

            #     outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]
    
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}

        scale = 0
        pred = outputs[("depth", 0, scale)]
        target = inputs[("depth_gt")]
        
        total_loss = self.compute_reprojection_loss(pred, target).mean()
        
        # for scale in self.scales:
        #     loss = 0
            # reprojection_losses = []

            # source_scale = 0

            # disp = outputs[("disp", scale)]
            # color = inputs[("color", 0, scale)]
            # target = outputs[("depth", 0, source_scale)]
            # target = inputs[("color", 0, source_scale)]

            # for frame_id in self.frame_ids[1:]:
            #     pred = outputs[("color", frame_id, scale)]
            #     reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            # reprojection_losses = torch.cat(reprojection_losses, 1)

            # identity_reprojection_losses = []
            # for frame_id in self.frame_ids[1:]:
            #     pred = inputs[("color", frame_id, source_scale)]
            #     identity_reprojection_losses.append(
            #         # if camera does not move, pred and target are the same, so that loss=0
            #         self.compute_reprojection_loss(pred, target))

            # identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            # # save both images, and do min all at once below
            # identity_reprojection_loss = identity_reprojection_losses

            # reprojection_loss = reprojection_losses

            # # add random numbers to break ties
            # identity_reprojection_loss += torch.randn(
            #     identity_reprojection_loss.shape, device=self.device) * 0.00001

            # # [not move, corretlymatch]
            # combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            # to_optimise, idxs = torch.min(combined, dim=1)

            # # true means corretly match, false means not move
            # outputs["identity_selection/{}".format(scale)] = (
            #     idxs > identity_reprojection_loss.shape[1] - 1).float()

            # loss += to_optimise.mean()
        
            # mean_disp = disp.mean(2, True).mean(3, True)
            # norm_disp = disp / (mean_disp + 1e-7)
            # smooth_loss = get_smooth_loss(norm_disp, color)

            # loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            # total_loss += loss
            # losses["loss/smooth_{}".format(scale)] = loss

        # if not self.disable_matcher:
        #     matcher_loss = 0
        #     for frame_id in self.frame_ids[1:]:
        #         # correspondences = self.matcher(inputs[('color', 0, source_scale)], inputs[('color', frame_id, source_scale)])
        #         correspondences = inputs[('correspondences', 0, frame_id)]
        #         for scale in self.scales:
        #             matcher_loss += compute_matcher_errors_from_correspondences(correspondences, outputs[("sample", frame_id, scale)],
        #                                                                         self.width, self.height, self.batch_size, self.device,
        #                                                                         confidence=self.confidence, delta=self.matcher_loss_delta)
        #     matcher_loss *= self.matcher_loss_alpha
        #     total_loss += matcher_loss
        #     losses["loss/matcher"] = matcher_loss
            
        # if self.ratio_consistency and outputs["do_shuffle"]:
        #     ratio_consistency_loss = 0
        #     for scale in self.scales:
        #         # ratio_consistency_loss += torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)]).mean()
        #         ratio_consistency_loss += self.compute_batch_image_shuffle_loss(outputs[("shuffle_disp", scale)], outputs[("disp", scale)], norm=self.ratio_consistency_normalization)
        #     total_loss += ratio_consistency_loss
        #     losses["loss/ratio_consistency"] = ratio_consistency_loss
        # elif self.ratio_consistency_crop and outputs["do_shuffle"]:
        #     ratio_consistency_crop_loss = 0
        #     for scale in self.scales:
        #         ratio_consistency_crop_loss_tmp = torch.abs(outputs[("shuffle_disp", scale)] - outputs[("disp", scale)])
        #         if self.ratio_consistency_normalization:
        #             ratio_consistency_crop_loss_tmp /= (outputs[("shuffle_disp", scale)] + outputs[("disp", scale)])
        #         if self.ratio_consistency_scales_normalization:
        #             ratio_consistency_crop_loss_tmp /= (2 ** scale)
        #         ratio_consistency_crop_loss += self.compute_random_batch_image_shuffle_loss(ratio_consistency_crop_loss_tmp, inputs["crop_info"] // 2**scale).mean()
        #     ratio_consistency_crop_loss *= self.weight_ratio_consistency_crop
        #     total_loss += ratio_consistency_crop_loss
        #     losses["loss/ratio_consistency_crop"] = ratio_consistency_crop_loss

        # if self.geometry_loss:
        #     geometry_loss = 0
        #     for scale in self.scales:
        #         for frame_id in self.frame_ids[1:]:
        #             if self.geometry_loss_disp_mode:
        #                 geometry_loss += torch.abs(outputs[("computed_disp", frame_id, scale)] - outputs[("sampled_disp", frame_id, scale)]).mean()
        #             else:
        #                 geometry_loss += (torch.abs(outputs[("computed_depth", frame_id, scale)] - outputs[("sampled_depth", frame_id, scale)]) / (outputs[("computed_depth", frame_id, scale)] + outputs[("sampled_depth", frame_id, scale)])).mean() / (2 ** scale)
        #     geometry_loss *= 0.1
        #     total_loss += geometry_loss
        #     losses["loss/geometry_loss"] = geometry_loss
                        

        # total_loss /= len(self.scales)
        losses["loss"] = total_loss
        return losses

    def compute_random_batch_image_shuffle_loss(self, l1, crop_info):
        b, _, h, w = l1.shape
        x1, y1, x1p, y1p, patch_width, patch_height, _, _ = crop_info
        mask = torch.ones_like(l1)
        min_x1p = max(x1p - 1, 0)
        max_x1p = min(x1p + patch_width + 1, w)
        min_y1p = max(y1p - 1, 0)
        max_y1p = min(y1p + patch_height + 1, h)
        mask[:b//2, :, min_y1p:min_y1p+2, min_x1p:max_x1p] = 0 # top
        mask[:b//2, :, max_y1p-2:max_y1p, min_x1p:max_x1p] = 0 # botton
        mask[:b//2, :, min_y1p:max_y1p, min_x1p:min_x1p+2] = 0 # left
        mask[:b//2, :, min_y1p:max_y1p, max_x1p-2:max_x1p] = 0 # right

        min_x1 = max(x1 - 1, 0)
        max_x1 = min(x1 + patch_width + 1, w)
        min_y1 = max(y1 - 1, 0)
        max_y1 = min(y1 + patch_height + 1, h)
        mask[b//2:, :, min_y1:min_y1+2, min_x1:max_x1] = 0 # top
        mask[b//2:, :, max_y1-2:max_y1, min_x1:max_x1] = 0 # botton
        mask[b//2:, :, min_y1:max_y1, min_x1:min_x1+2] = 0 # left
        mask[b//2:, :, min_y1:max_y1, max_x1-2:max_x1] = 0 # right

        return l1 * mask
    
    def compute_batch_image_shuffle_loss(self, pred, target, norm=False):
        mask = torch.ones_like(pred)
        b, _, h, w = pred.shape
        mask[:, :, :, w//2-1:w//2+1] = 0
        mask[:, :, h//2-1:h//2+1, :] = 0
        if norm:
            return ((torch.abs(pred - target) / (pred + target)) * mask).mean()
        else:
            return torch.abs((pred - target) * mask).mean()
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    @torch.no_grad()
    def compute_depth_losses(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_losses = {}
        depth_gt = inputs["depth_gt"]
        b, c, gt_h, gt_w = depth_gt.shape
        mask = depth_gt > 0
        
        depth_pred = outputs[("depth", 0, 0)]
        # depth_pred = F.interpolate(
        #     depth_pred, [gt_h, gt_w], mode="bilinear", align_corners=False)
        depth_pred = depth_pred.detach()

        
        depth_gt_flatten = depth_gt.view(self.batch_size, -1)
        depth_pred_flatten = depth_pred.view(self.batch_size, -1)
        mask_flatten = mask.view(self.batch_size, -1)
        
        med_gt, _ = torch.masked_fill(depth_gt_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)
        med_pred, _ = torch.masked_fill(depth_pred_flatten, ~mask_flatten, float("nan")).nanmedian(dim=1, keepdim=True)

        ratios = med_gt / med_pred
        avg = torch.mean(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios/med)

        depth_losses['ratio/mean'] = np.array(avg.cpu())
        depth_losses['ratio/med'] = np.array(med.cpu())
        depth_losses['ratio/std'] = np.array(std.cpu())
        
        depth_pred *= ratios[..., None, None]
        
        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]

        depth_pred = torch.clamp(depth_pred, min=self.min_gt_depth, max=self.max_gt_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            depth_losses[metric] = np.array(depth_errors[i].cpu())
            
        return depth_losses

```

### 文件: `trainer.py`

```py
import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter

from .utils import *
from .datasets.gastro_dataset import get_data_loaders
from . import meters


class Trainer():
    def __init__(self, cfgs, model):
        self.cfgs = cfgs
        self.log_path = Path(cfgs.get('log_dir', 'results')) / cfgs.get('model_name',
                                                                        datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        self.device = cfgs.get('device', 'cpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.batch_size = cfgs.get('batch_size', 64)
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.start_epoch = cfgs.get('start_epoch', 0)
        self.load_weights_folder = cfgs.get('load_weights_folder', None)
        self.mypretrain = cfgs.get('mypretrain', None)
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.run_val = cfgs.get('run_val', True)
        self.run_test = cfgs.get('run_test', False)
        # geometry loss
        self.geometry_loss = cfgs.get('geometry_loss', False)
        self.supervised = cfgs.get('supervised', False)

        # --- 新增早停相关配置 ---
        self.early_stop_patience = cfgs.get('early_stop_patience', 5)  # 容忍度
        self.early_stop_metric = cfgs.get('early_stop_metric', 'de/abs_rel')  # 监控指标
        self.best_val_metric = float('inf')
        self.patience_counter = 0
        self.best_epoch = -1

        self.model = model(cfgs)
        self.model.trainer = self

        data_loaders = get_data_loaders(cfgs)
        self.train_loader = data_loaders["train_loader"]
        if self.run_val:
            self.val_loader = data_loaders["val_loader"]
            self.val_iter = iter(self.val_loader)
        if self.run_test:
            self.test_loader = data_loaders["test_loader"]
        info_dict = data_loaders["info_dict"]

        cfgs.update(info_dict)
        self.num_total_steps = info_dict['num_total_steps']
        if self.start_epoch > 0:
            self.num_total_steps -= self.num_total_steps / self.num_epochs * self.start_epoch
        self.metrics_trace = meters.MetricsTrace()
        self.metric_str_exclude = cfgs.get('metric_str_exclude', [])
        self.make_metrics = lambda m=None: meters.StandardMetrics(m, self.metric_str_exclude)

    def load_checkpoint(self, custom_path=None):
        """支持加载指定路径或默认配置路径的权重"""
        if custom_path is not None:
            # 临时修改模型中的加载路径并加载
            original_folder = self.model.load_weights_folder
            self.model.load_weights_folder = custom_path
            self.model.load_model()
            self.model.load_weights_folder = original_folder
            return

        if self.mypretrain is not None:
            self.model.load_pretrain()

        if self.load_weights_folder is not None:
            self.model.load_model()

    def save_checkpoint(self, is_best=False):
        """保存模型权重"""
        folder_name = "weights_best" if is_best else "weights_{}".format(self.epoch)
        save_folder = os.path.join(self.log_path, "models", folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if not is_best and self.epoch + 1 == self.num_epochs:
            last_folder = os.path.join(self.log_path, "models", "weights_last")
            if os.path.exists(last_folder): os.remove(last_folder)
            os.symlink(folder_name, last_folder)

        for net_name in self.model.network_names:
            save_path = os.path.join(save_folder, "{}.pth".format(net_name))
            to_save = getattr(self.model, net_name).state_dict()
            if net_name == "net_depth_encoder":
                to_save['height'] = self.model.height
                to_save['width'] = self.model.width
            torch.save(to_save, save_path)

        for optim_name in self.model.optimizer_names:
            save_path = os.path.join(save_folder, "{}.pth".format(optim_name))
            torch.save(getattr(self.model, optim_name).state_dict(), save_path)

    def train(self):
        """主训练流程"""
        self.metrics_trace.reset()
        self.start_time = time.time()
        self.train_iter_per_epoch = len(self.train_loader)
        if self.run_val:
            self.val_iter_per_epoch = len(self.val_loader)
        if self.run_test:
            self.test_iter_per_epoch = len(self.test_loader)

        self.model.to_device(self.device)
        self.model.init_optimizers()

        self.load_checkpoint()

        self.writers = {}
        for mode in ["train", "val", "test"]:
            self.writers[mode] = SummaryWriter(Path(self.log_path) / "logs" / mode)

        print("Training model named:\n  ", self.model.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        print(f"{self.model.model_name}: optimizing to {self.num_epochs} epochs")

        if self.start_epoch > 0:
            for _ in range(self.start_epoch):
                for scheduler_name in self.model.scheduler_names:
                    getattr(self.model, scheduler_name).step()

        for self.epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()

            # --- 1. 训练阶段 ---
            metrics_train = self.run_epoch(self.train_loader, self.epoch, is_train=True)
            self.metrics_trace.append("train", metrics_train)

            # --- 2. 验证阶段 ---
            if self.run_val:
                with torch.no_grad():
                    metrics_val = self.run_epoch(self.val_loader, self.epoch, is_train=False, is_val=True)
                    self.metrics_trace.append("val", metrics_val)

                # 早停逻辑检查
                val_stats = metrics_val.get_data_dict()
                current_val_metric = val_stats.get(self.early_stop_metric, val_stats.get('loss', 0))

                if current_val_metric < self.best_val_metric:
                    self.best_val_metric = current_val_metric
                    self.best_epoch = self.epoch
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                    print(f"✨ 发现最佳模型! ({self.early_stop_metric}: {self.best_val_metric:.6f})")
                else:
                    self.patience_counter += 1
                    print(f"⚠️  验证指标未改善 (耐心计数: {self.patience_counter}/{self.early_stop_patience})")

            # --- 3. 测试阶段 ---
            if self.run_test:
                with torch.no_grad():
                    metrics_test = self.run_epoch(self.test_loader, self.epoch, is_train=False, is_test=True)
                    self.metrics_trace.append("test", metrics_test)

            # 定期保存
            if (self.epoch + 1) % self.save_checkpoint_freq == 0 or (self.epoch + 1) == self.num_epochs:
                self.save_checkpoint()

            self.metrics_trace.save(os.path.join(self.log_path, 'metrics.json'))

            epoch_duration = time.time() - epoch_start_time
            print(f"✅ Epoch {self.epoch} duration: {epoch_duration:.2f} seconds.\n")

            # --- 4. 触发早停 ---
            if self.patience_counter >= self.early_stop_patience:
                print(f"🚨 早停触发！在epoch {self.epoch} 停止训练。")
                print(f"最佳验证指标: {self.best_val_metric:.6f} (epoch {self.best_epoch})")

                # 恢复最佳权重
                best_weights_path = os.path.join(self.log_path, "models", "weights_best")
                print(f"恢复最佳模型权重 (epoch {self.best_epoch})...")
                self.load_checkpoint(custom_path=best_weights_path)

                for net_name in self.model.network_names:
                    print(f"✅ 恢复 {net_name} 权重")
                print("🎯 最佳模型权重已恢复")
                print(f"🎯 训练因早停而提前结束于epoch {self.epoch + 1}")
                break

        print(f"Training completed after {self.epoch + 1} epochs.")

    def run_epoch(self, loader, epoch=0, is_train=True, is_val=False, is_test=False):
        """运行单个 Epoch"""
        metrics = self.make_metrics()

        if is_train:
            self.model.set_train()
            mode = "TRAIN"
            iter_per_epoch = self.train_iter_per_epoch
            for scheduler_name in self.model.scheduler_names:
                getattr(self.model, scheduler_name).step()
        elif is_val:
            self.model.set_eval()
            mode = "VAL"
            iter_per_epoch = self.val_iter_per_epoch
        elif is_test:
            self.model.set_eval()
            mode = "TEST"
            iter_per_epoch = self.test_iter_per_epoch
        else:
            raise NotImplementedError

        for batch_idx, inputs in enumerate(loader):
            before_op_time = time.time()

            outputs, losses = self.model.forward(inputs)

            if is_train:
                self.model.backward(losses)

            duration = time.time() - before_op_time
            metrics.update(losses, self.batch_size)

            # 处理指标更新 (如果有GT深度)
            if "depth_gt" in inputs:
                depth_losses = self.model.compute_depth_losses(inputs, outputs)
                metrics.update(depth_losses, self.batch_size)
                losses.update(depth_losses)

            # 日志记录 (Tensorboard)
            total_iter = batch_idx + epoch * iter_per_epoch
            if self.use_logger and (total_iter % self.log_freq == 0):
                if is_train:
                    iter_sofar = batch_idx + (epoch - self.start_epoch) * self.train_iter_per_epoch
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data, iter_sofar)
                self.log(mode.lower(), inputs, outputs, losses, total_iter)

            # --- 实时后台打印 ---
            if batch_idx % 10 == 0 or batch_idx == iter_per_epoch - 1:
                fps = self.batch_size / duration if duration > 0 else 0
                loss_str = " ".join(
                    [f"{k}: {v:.5f}" for k, v in losses.items() if "loss" in k or "de/" in k or "da/" in k])
                print(f"📊 [{mode:^6}] 第{epoch}个epoch | 进度: {batch_idx:04d}/{iter_per_epoch:04d} | "
                      f"耗时: {duration:.2f}s | {fps:7.1f}Hz      {loss_str}")

        # --- Epoch 结束时的总结打印 ---
        final_metrics = metrics.get_data_dict()
        if self.use_logger:
            for k, v in final_metrics.items():
                self.writers[mode.lower()].add_scalar(f'Metrics/{k}', v, epoch)

        print("=" * 60)
        print(f"📈 [{mode} 指标] 第{epoch}个epoch：")
        print("─" * 60)
        for k, v in final_metrics.items():
            print(f"  {k:<25} :   {v:.6f}")
        print("=" * 60)

        return metrics

    def log(self, mode, inputs, outputs, losses, step):
        """写入 Tensorboard 事件"""
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, step)

        for j in range(min(4, self.batch_size)):
            for s in self.model.scales:
                for frame_id in self.model.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), step)

                if not self.model.disable_automasking:
                    if "identity_selection/{}".format(s) in outputs:
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j][None, ...], step)

                if self.geometry_loss:
                    for frame_id in self.model.frame_ids[1:]:
                        if ("computed_depth", frame_id, s) in outputs:
                            writer.add_image(
                                "computed_depth_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(outputs[("computed_depth", frame_id, s)][j]), step)
                            writer.add_image(
                                "sampled_depth_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(outputs[("sampled_depth", frame_id, s)][j]), step)

    def log_time(self, batch_idx, duration, loss, step):
        """打印标准时间日志（可选保留）"""
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / step - 1.0) * time_sofar if step > 0 else 0

        lr_d = self.model.optimizer_depth.param_groups[0]['lr']
        if self.supervised:
            print_string = "epoch {:>3} | lr {:.6f} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f}"
            print(print_string.format(self.epoch, lr_d, batch_idx, samples_per_sec, loss))
        else:
            lr_p = self.model.optimizer_pose.param_groups[0]['lr']
            print_string = "epoch {:>3} | lr_d {:.6f} | lr_p {:.6f} | batch {:>6} | loss: {:.5f}"
            print(print_string.format(self.epoch, lr_d, lr_p, batch_idx, loss))
```

### 文件: `utils.py`

```py
from __future__ import absolute_import, division, print_function
import glob
import os
import random
from functools import partial

import numpy as np
import torch
import yaml



def setup_runtime(args):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""
    
    # Setup CUDA
    cuda_device_id = args.gpu
    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        #print("CUDA_VISIBLE_DEVICES {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # avoid bad file descripter
    torch.multiprocessing.set_sharing_strategy('file_system')

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    cfgs['device'] = cuda_device_id

    print(f"Environment: GPU {cuda_device_id} seed {args.seed} number of workers {args.num_workers}")
    
    return cfgs

unsqueezer = partial(torch.unsqueeze, dim=0)

def map_fn(batch, fn):
    if isinstance(batch, dict):
        for k in batch.keys():
            batch[k] = map_fn(batch[k], fn)
        return batch
    elif isinstance(batch, list):
        return [map_fn(e, fn) for e in batch]
    else:
        return fn(batch)


def to(data, device):
    if isinstance(data, dict):
        return {k: to(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to(v, device) for v in data]
    else:
        return data.to(device)

def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfgs):
    print(f"Saving configs to {path}")
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)
```

### 目录: `depthnet\datasets`

#### 文件: `__init__.py`

```py
from .gastro_dataset import C3VDDataset, SimcolDataset, NYUDataset, TestNYUDataset
from .kitti_dataset import KITTIRAWDataset
```

#### 文件: `gastro_dataset.py`

```py
import torch
import random
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import os

from PIL import Image, ImageFile
import skimage.transform
import numpy as np
import PIL.Image as pil
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import collections

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def pil_depth_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('I')


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, collections.abc.Mapping):  # Some custom condition
        processed_batch = {}
        for key in elem:
            if "correspondences" in key:
                processed_batch[key] = [d[key] for d in batch]
            else:
                processed_batch[key] = default_collate([d[key] for d in batch])
        try:
            return elem_type(processed_batch)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return processed_batch
    else:  # Fall back to `default_collate`
        return default_collate(batch)


def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 64)
    num_epochs = cfgs.get('num_epochs', 30)
    num_workers = cfgs.get('num_workers', 8)
    data_path = cfgs.get('data_path')
    height = cfgs.get('height', 256)
    width = cfgs.get('width', 320)
    frame_ids = cfgs.get('frame_ids', [0, -1, 1])
    num_scales = len(cfgs.get('scales', [0, 1, 2, 3]))
    dataset = globals().get(cfgs.get('dataset', C3VDDataset))
    split = cfgs.get('split')
    fpath = os.path.join("splits", split, "{}_files.txt")
    img_ext = '.png' if cfgs.get('png', False) else '.jpg'
    load_depth = cfgs.get('load_depth', False)

    train_filenames = readlines(fpath.format("train"))
    matcher_result_load_train = cfgs.get('matcher_result_train', None)
    if matcher_result_load_train:
        matcher_result_load_train = np.load(matcher_result_load_train, allow_pickle=True).all()

    train_dataset = dataset(data_path, train_filenames, matcher_result_load_train,
                            height, width, frame_ids, num_scales,
                            is_train=True, img_ext=img_ext, load_depth=load_depth, local_crop=True,
                            patch_reshuffle=True)

    train_loader = DataLoader(
        train_dataset, batch_size, True, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    info_dict = {"num_total_steps": len(train_filenames) // batch_size * num_epochs}

    loaders_dict = {"train_loader": train_loader, "info_dict": info_dict}

    val_dataset, test_dataset = ([], [])

    if cfgs.get('run_val', False):
        val_filenames = readlines(fpath.format("val"))
        matcher_result_load_val = cfgs.get('matcher_result_val', None)
        if matcher_result_load_val:
            matcher_result_load_val = np.load(matcher_result_load_val, allow_pickle=True).all()

        val_dataset = dataset(data_path, val_filenames, matcher_result_load_val,
                              height, width, frame_ids, num_scales,
                              is_train=False, img_ext=img_ext, load_depth=load_depth, local_crop=True,
                              patch_reshuffle=True)

        val_loader = DataLoader(
            val_dataset, batch_size, True, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=False, drop_last=True)

        loaders_dict["val_loader"] = val_loader

    if cfgs.get('run_test', False):
        test_filenames = readlines(fpath.format("test"))
        matcher_result_load_test = cfgs.get('matcher_result_test', None)
        if matcher_result_load_test:
            matcher_result_load_test = np.load(matcher_result_load_test, allow_pickle=True).all()

        test_dataset = dataset(data_path, test_filenames, matcher_result_load_test,
                               height, width, frame_ids, num_scales,
                               is_train=False, img_ext=img_ext, load_depth=load_depth, local_crop=True,
                               patch_reshuffle=True)

        test_loader = DataLoader(
            test_dataset, batch_size, True, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=False, drop_last=True)

        loaders_dict["test_loader"] = test_loader

    print("Using split:\n  ", split)
    print("There are {:d} training items, {:d} validation items and {:d} tesing items\n".format(
        len(train_dataset), len(val_dataset), len(test_dataset)))

    return loaders_dict


class MonoDataset(torch.utils.data.Dataset):
    """MonoDataset

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 resize_ratio=[1.2, 2.0],
                 local_crop=False,
                 split_ratio=[0.1, 0.9],
                 patch_reshuffle=False,
                 is_train=False,
                 img_ext='.jpg',
                 load_depth=False,
                 load_pose=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # self.interp = Image.ANTIALIAS
        self.interp = torchvision.transforms.InterpolationMode.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext
        self.load_depth = load_depth
        self.load_pose = load_pose

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.resize_ratio_lower = resize_ratio[0]
        self.resize_ratio_upper = resize_ratio[1]
        self.local_crop = local_crop
        self.split_ratio_lower = split_ratio[0]
        self.split_ratio_upper = split_ratio[1]
        self.patch_reshuffle = patch_reshuffle

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        """
        for k in list(inputs):
            if "color" in k:
                n, im, _ = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        """

        if self.local_crop:
            resize_ratio = (
                                       self.resize_ratio_upper - self.resize_ratio_lower) * random.random() + self.resize_ratio_lower
            height_re = int(self.height * resize_ratio)
            width_re = int(self.width * resize_ratio)
            w0 = int((width_re - self.width) * random.random())
            h0 = int((height_re - self.height) * random.random())
            self.resize_local = transforms.Resize((height_re, width_re), interpolation=self.interp)
            box = (w0, h0, w0 + self.width, h0 + self.height)
            gridx, gridy = np.meshgrid(np.linspace(-1, 1, width_re), np.linspace(-1, 1, height_re))
            gridx = torch.from_numpy(gridx)
            gridy = torch.from_numpy(gridy)
            grid = torch.stack([gridx, gridy], dim=0)
            inputs[("grid_local")] = grid[:, h0: h0 + self.height, w0: w0 + self.width].clone()
            inputs[("ratio_local")] = torch.tensor([resize_ratio])

        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    if self.local_crop:
                        if i == 0:
                            inputs[(n + "_local", im, i)] = self.resize_local(inputs[(n, im, -1)]).crop(box)
                        else:
                            inputs[(n + "_local", im, i)] = self.resize[i](inputs[(n + "_local", im, i - 1)])
                if self.patch_reshuffle:
                    if im == 0:
                        ### Split-Permute as depicted in paper (vertical + horizontal)
                        img = inputs[(n, im, 0)]
                        newimg = img.copy()
                        ratio_x = random.random() * (
                                    self.split_ratio_upper - self.split_ratio_lower) + self.split_ratio_lower
                        ratio_y = random.random() * (
                                    self.split_ratio_upper - self.split_ratio_lower) + self.split_ratio_lower
                        w_i = int(self.width * ratio_x)
                        patch1 = img.crop((0, 0, w_i, self.height)).copy()
                        patch2 = img.crop((w_i, 0, self.width, self.height)).copy()
                        newimg.paste(patch2, (0, 0))
                        newimg.paste(patch1, (self.width - w_i, 0))
                        h_i = int(self.height * ratio_y)
                        patch1 = newimg.crop((0, 0, self.width, h_i)).copy()
                        patch2 = newimg.crop((0, h_i, self.width, self.height)).copy()
                        newimg.paste(patch2, (0, 0))
                        newimg.paste(patch1, (0, self.height - h_i))
                        inputs[(n + "_reshuffle", im, 0)] = newimg
                        inputs[("split_xy")] = torch.tensor([self.width - w_i, self.height - h_i])

                        ### Split-Permute (vertical or horizontal, randomly choose one)
                        # img = inputs[(n, im, 0)]
                        # newimg = img.copy()
                        # ratio = random.random() * (self.split_ratio_upper - self.split_ratio_lower) + self.split_ratio_lower
                        # if random.random() > 0.5:
                        #     w_i = int(self.width * ratio)
                        #     patch1 = img.crop((0, 0, w_i, self.height)).copy()
                        #     patch2 = img.crop((w_i, 0, self.width, self.height)).copy()
                        #     newimg.paste(patch2, (0, 0))
                        #     newimg.paste(patch1, (self.width-w_i, 0))
                        #     inputs[(n + "_reshuffle", im, 0)] = newimg
                        #     inputs[("split_xy")] = torch.tensor([self.width-w_i, 0])
                        # else:
                        #     h_i = int(self.height * ratio)
                        #     patch1 = img.crop((0, 0, self.width, h_i)).copy()
                        #     patch2 = img.crop((0, h_i, self.width, self.height)).copy()
                        #     newimg.paste(patch2, (0, 0))
                        #     newimg.paste(patch1, (0, self.height-h_i))
                        #     inputs[(n + "_reshuffle", im, 0)] = newimg
                        #     inputs[("split_xy")] = torch.tensor([0, self.height-h_i])
        for k in list(inputs):
            f = inputs[k]
            if "color" in k[0]:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Return a single training item from the dataset as a dictionary

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)      for raw colour images,
            ("K", scale) or ("inv_K", scale)    for camera intrinsics,
            "stereo_T"                          for camera extrinscis

        <frame_id> is an integer (e.g. 0, -1, or 1) representing the emporal step relative to 'index'
        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        inputs.update(self.load_extra(do_flip, index))

        for i in self.frame_idxs:
            inputs[('color', i, -1)] = self.get_color(folder, line[2 + i], do_flip)
            if self.load_depth and i == 0:  # only frame 0 need depth_gt
                depth = self.get_depth(folder, line[2 + i], do_flip).resize((self.width, self.height))
                depth = self.postprocess_depth(depth)
                inputs[('depth_gt')] = depth

            if self.load_pose:
                inputs[('pose_gt', i, 0)] = self.get_pose(folder, line[2 + i], do_flip)

        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            def color_aug(img):
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = F.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = F.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = F.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = F.adjust_hue(img, hue_factor)
                return img
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        return inputs

    def get_color(self, folder, frame_index_str, do_flip):
        raise NotImplementedError

    def get_depth(self, folder, frame_index_str, do_flip):
        raise NotImplementedError

    def get_pose(self, folder, frame_index_str, do_flip):
        raise NotImplementedError

    def postprocess_depth(self, depth):
        return torch.from_numpy(np.array(depth))

    def load_extra(self, do_flip, index):
        return {}


class C3VDDataset(MonoDataset):
    def __init__(self,
                 data_path,
                 filenames,
                 correspondences,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 load_depth=False,
                 load_pose=False,
                 local_crop=False,
                 patch_reshuffle=False):
        super(C3VDDataset, self).__init__(data_path, filenames, height, width, frame_idxs,
                                          num_scales, is_train=is_train, img_ext=img_ext, load_depth=load_depth,
                                          load_pose=load_pose, local_crop=local_crop, patch_reshuffle=patch_reshuffle)
        self.K = np.array([[0.56959306, 0, 0.5, 0],
                           [0, 0.71185083, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1350, 1080)  # reshape the whole dataset before hand. raw shape is [w=1350, h=1080]
        self.correspondences = correspondences  # kwargs['correspondences']
        self.loader_depth = pil_depth_loader

    def get_color(self, folder, frame_index_str, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index_str))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index_str):
        f_str = "{}_color{}".format(frame_index_str, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_depth_path(self, folder, frame_index_str):
        f_str = "{}_depth{}".format(frame_index_str, ".tiff")
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_depth(self, folder, frame_index_str, do_flip):
        depth = self.loader_depth(self.get_depth_path(folder, frame_index_str))

        if do_flip:
            depth = depth.transpose(pil.FLIP_LEFT_RIGHT)
        # depth = np.array(depth)/(2**16-1)
        return depth

    def get_pose(self, folder, frame_index_str, do_flip):
        pose_path = os.path.join(self.data_path, folder, "pose.txt")
        with open(pose_path, 'r') as f:
            lines = f.read().splitlines()
        pose = lines[int(frame_index_str)].split(",")
        pose = np.array(pose, dtype=float)
        pose = pose.reshape(4, 4)
        return pose

    def postprocess_depth(self, depth):
        return torch.from_numpy(np.array(depth) / (2 ** 16 - 1))[None, ...]

    def load_extra(self, do_flip, index):
        inputs = {}
        if self.correspondences:
            if do_flip:
                flip_str = 'do_flip'
            else:
                flip_str = 'no_flip'

            for idx, fid in [[0, -1], [1, 1]]:
                source_fid = 0
                inputs[("correspondences", source_fid, fid)] = self.correspondences[flip_str][index][idx]
        return inputs


class SimcolDataset(C3VDDataset):
    def __init__(self, *args, **kwargs):
        super(SimcolDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.5, 0, 0.5, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (475, 475)  # reshape the whole dataset before hand

    def get_image_path(self, folder, frame_index_str):
        f_str = "{}{}".format(frame_index_str, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_depth_path(self, folder, frame_index_str):
        f_str = "{}{}".format(frame_index_str.replace("FrameBuffer", "Depth"), ".png")
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def postprocess_depth(self, depth):
        return torch.from_numpy(np.array(depth) / (255 * 256))[None, ...]


class NYUDataset(C3VDDataset):
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)
        # 518.8579 0 325.58245 - 41
        # 0 519.46961 253.73617 - 45
        # 0 0 1

        # 518.8579 0 284.58245
        # 0 519.46961 208.73617
        # 0 0 1

        self.K = np.array([[518.8579 / 560, 0, 284.58245 / 560, 0],
                           [0, 519.46961 / 426, 208.73617 / 426, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (560, 426)  # reshape the whole dataset before hand

    def get_image_path(self, folder, frame_index_str):
        f_str = "{}{}".format(frame_index_str, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_depth_path(self, folder, frame_index_str):
        f_str = "{}{}".format("depth/" + frame_index_str, ".png")
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def postprocess_depth(self, depth):
        return torch.from_numpy(np.array(depth) / 5000)[None, ...]


class TestNYUDataset(NYUDataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs=[-1],
                 num_scales=1,
                 is_train=False,
                 img_ext='.png',
                 load_depth=False,
                 load_pose=False):
        super(NYUDataset, self).__init__(data_path, filenames, None, height, width, frame_idxs,
                                         num_scales, is_train=is_train, img_ext=img_ext, load_depth=load_depth,
                                         load_pose=load_pose)

```

#### 文件: `kitti_dataset.py`

```py
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import skimage.transform
import numpy as np
import PIL.Image as pil
from PIL import Image  # using pillow-simd for increased speed
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.utils.data as data
import torchvision

from depthnet.kitti_utils import generate_depth_map

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 load_depth=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # self.interp = Image.ANTIALIAS
        self.interp = torchvision.transforms.InterpolationMode.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = load_depth #self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
            def color_aug(img):
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = F.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = F.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = F.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = F.adjust_hue(img, hue_factor)
                return img
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, data_path, filenames, correspondences, height, width, frame_idxs,
                 num_scales, is_train=False, img_ext='.jpg', load_depth=False):
        super(KITTIRAWDataset, self).__init__(data_path, filenames, height, width, frame_idxs,
                 num_scales, is_train=False, img_ext=img_ext, load_depth=load_depth)
        self.correspondences = correspondences

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}{}".format(frame_index, ".png")
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        depth_path = image_path.replace("imgs", "depths")
        
        depth_gt = np.array(Image.open(depth_path)).astype(np.float32) / 256
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    # def get_depth(self, folder, frame_index, side, do_flip):
    #     calib_path = os.path.join(self.data_path, folder.split("/")[0])

    #     velo_filename = os.path.join(
    #         self.data_path,
    #         folder,
    #         "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

    #     depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
    #     # depth_gt = skimage.transform.resize(
    #     #     depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
    #     depth_gt = skimage.transform.resize(
    #         depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

    #     if do_flip:
    #         depth_gt = np.fliplr(depth_gt)

    #     return depth_gt
    
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        
        if self.correspondences:
            if do_flip:
                flip_str = 'do_flip'
            else:
                flip_str = 'no_flip'
            for idx, fid in [[0, -1], [1, 1]]:
                source_fid = 0
                inputs[("correspondences", source_fid, fid)] = self.correspondences[flip_str][index][idx]
            


        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
            def color_aug(img):
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = F.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = F.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = F.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = F.adjust_hue(img, hue_factor)
                return img
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path

    
class KITTIOdomGreyDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomGreyDataset, self).__init__(*args, **kwargs)
        self.side_map = {"2": 0, "3": 1, "l": 0, "r": 1}

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path
    
    def check_depth(self):
        pass
    

class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

```

### 目录: `depthnet\networks`

#### 文件: `ASSA.py`

```py
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import repeat

# 论文：Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration, CVPR 2024.
# 论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v

# Adaptive Sparse Self-Attention (ASSA)
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads=8, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2  # b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    # Instantiate the WindowAttention_sparse class
    dim = 256  # Dimension of input features
    win_size = (8,10)  # Window size(H, W)
    # Create an instance of the WindowAttention_sparse module
    window_attention_sparse = WindowAttention_sparse(dim, win_size)
    C = dim
    input = torch.randn(12, 8 * 10, C)#输入B H W
    # Forward pass
    output = window_attention_sparse(input)

    # Print input and output size
    print(input.size())
    print(output.size())

```

#### 文件: `FMBConv.py`

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution( ECCV 2024 )
# 论文地址：https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ren_The_Ninth_NTIRE_2024_Efficient_Super-Resolution_Challenge_Report_CVPRW_2024_paper.pdf

# partial convolution-based feed-forward network
class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x

class LightPCFN(nn.Module):
    def __init__(self, dim, growth_rate=1.5, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)

        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # depthwise conv 替代 3x3 conv
        self.conv_1 = nn.Conv2d(p_dim, p_dim, kernel_size=3, padding=1, groups=p_dim, bias=False)

        # 投影回原始维度
        self.conv_2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv_0(x)
        x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
        x1 = self.conv_1(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv_2(x)



class PEC_SMFA(nn.Module):
    """
    参数高效的坐标感知自调制特征聚合 (Parameter-Efficient Coordinate‑aware Self‑Modulation Feature Aggregation)
    输入:
        x: Tensor of shape (B, C, H, W)
    输出:
        modulated: Tensor of same shape, 通过空间感知的调制图 M 与原特征逐点相乘得到
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        # 深度可分离 1D 卷积：groups=channels，参数量 ≈ 3*C
        padding = kernel_size // 2
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 1) 双向全局平均池化
        #   X_h: [B, C, H, 1], X_w: [B, C, 1, W]
        x_h = F.adaptive_avg_pool2d(x, (H, 1)).view(B, C, H)
        x_w = F.adaptive_avg_pool2d(x, (1, W)).view(B, C, W)

        # 2) 共享 1D 深度可分离卷积，捕捉跨维度依赖
        #   y_h: [B, C, H], y_w: [B, C, W]
        y_h = self.conv1d(x_h)
        y_w = self.conv1d(x_w)

        # 3) 恢复空间维度并拼合注意力图
        #   y_h → [B, C, H, 1], y_w → [B, C, 1, W]
        y_h = y_h.unsqueeze(-1)
        y_w = y_w.unsqueeze(-2)

        #   M: [B, C, H, W]
        M = self.sigmoid(y_h + y_w)

        # 4) 特征自调制
        return x * M


class FMBPlusPlus(nn.Module):
    def __init__(self, dim, out_dim, ffn_scale=2.0):
        super().__init__()
        self.smfa = PEC_SMFA(dim)
        #self.pcfn = LightPCFN(dim, ffn_scale)

        # 可学习门控控制融合程度
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, out_dim, kernel_size=1)

    def forward(self, x):
        smfa_out = self.smfa(F.normalize(x))
        #pcfn_out = self.pcfn(F.normalize(smfa_out))

        out = smfa_out
        g = self.gate(x)
        o = g * out + (1 - g) * x
        o = self.proj(o)
        return o




if __name__ == '__main__':
    input_shape = (1, 36, 64, 64)
    input = torch.randn(input_shape)

    # 实例化FMB类
    block = FMB(dim=36)

    # 将输入张量传入FMB实例
    output = block(input)

    # 打印输入和输出的形状
    print(input.size())
    print(output.size())
```

#### 文件: `LAE.py`

```py
import torch
import torch.nn as nn
from einops import rearrange
#from .LPA import LPA
import torch.nn.functional as F
from .modifyppm import ModifyPPM
from .LPA1 import LPA
# 论文地址：https://arxiv.org/pdf/2408.14087
# 论文：LSM-YOLO: A Compact and Effective ROI Detector for Medical Detection


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class LAE(nn.Module):
    # Light-weight Adaptive Extraction
    def __init__(self, ch, group=16) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )

        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))
        self.lpa = LPA(256)
        self.ppm = ModifyPPM(256,64,[3,6,9,12])

    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4

        att = rearrange(self.lpa(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)

        att = self.softmax(att)

        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        #x = self.ppm(x)
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        x = F.interpolate(x,scale_factor=2,mode="bilinear")
        return x


if __name__ == '__main__':

    input = torch.randn(12, 256, 8, 10) # B C H W
    block = LAE(ch=256)
    output = block(input)

    print(input.size())
    print(output.size())
```

#### 文件: `LDAM.py`

```py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

def lambda_init_fn(depth):
    # 根据深度动态调整初始λ，深度越深，差分权重相对越稳定
    if depth is None: depth = 1
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class LDAM(nn.Module):
    """
    两级差分注意力融合模块 (Two-level Differential Attention Fusion Module)
    用于肠道内窥镜深度估计的跳跃连接处。
    """
    def __init__(self, enc_dim, dec_dim, out_dim, n=49, block_depth=1, num_heads=8):
        super().__init__()
        self.n = n  # 全局视觉令牌数量
        self.num_heads = num_heads
        
        # 1. 特征拼接与对齐
        self.fusion_conv = nn.Sequential(
            nn.Linear(enc_dim + dec_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        head_dim = out_dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 2. 局部差分交互 (Local-level)
        # 使用DW-Conv模拟局部窗口内的特征差异提取
        self.local_dwc = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim)
        self.local_norm = RMSNorm(out_dim)
        
        # 3. 全局差分注意力 (Global-level)
        self.qkv = nn.Linear(out_dim, out_dim * 3)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(int(n**0.5), int(n**0.5)))
        
        # 视觉令牌偏置 (Positive/Negative Streams)
        self.e_pos = nn.Parameter(torch.randn(1, n, out_dim) * 0.02)
        self.e_neg = nn.Parameter(torch.randn(1, n, out_dim) * 0.02)
        
        # 差分系数 λ
        self.lambda_init = lambda_init_fn(block_depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim).normal_(0, 0.1))
        
        self.subln = RMSNorm(head_dim)
        self.proj = nn.Linear(out_dim, out_dim)
        
        # 4. 可学习通道门控分支
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 4, out_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, f_enc, f_dec):
        """
        f_enc: 编码器特征 [B, C1, H, W]
        f_dec: 解码器上采样特征 [B, C2, H, W]
        """
        B, C_e, H, W = f_enc.shape
        N = H * W
        
        # --- 步骤 1: 特征拼接与空间展平 ---
        # [B, C1+C2, H, W]
        x_cat = torch.cat([f_enc, f_dec], dim=1)
        x = x_cat.permute(0, 2, 3, 1).reshape(B, N, -1)
        x = self.fusion_conv(x) # [B, N, out_dim]
        
        # --- 步骤 2: 局部级差分响应 (Local-level) ---
        # 模拟局部窗口对黏膜、血管边缘的捕捉
        x_res = x.transpose(1, 2).reshape(B, -1, H, W)
        local_feat = self.local_dwc(x_res)
        # 显式计算差分响应: 原始特征 - 局部平滑特征
        local_diff = x_res - local_feat 
        local_out = self.local_norm(local_diff.permute(0, 2, 3, 1).reshape(B, N, -1))
        x = x + local_out
        
        # --- 步骤 3: 全局级差分注意力 (Global-level) ---
        c = x.shape[-1]
        num_heads = self.num_heads
        head_dim = c // num_heads
        
        qkv = self.qkv(x).reshape(B, N, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 全局蒸馏生成视觉令牌 t_tilde
        # [B, out_dim, h, w] -> [B, n, out_dim]
        t_tilde = self.pool(q.reshape(B, H, W, c).permute(0, 3, 1, 2))
        t_tilde = t_tilde.reshape(B, c, -1).permute(0, 2, 1)
        
        # 构建正负流: t_+ = t_tilde + e^+, t_- = t_tilde + e^-
        t_pos = t_tilde + self.e_pos
        t_neg = t_tilde + self.e_neg
        
        # 维度变换
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        t_pos = t_pos.reshape(B, self.n, num_heads, head_dim).permute(0, 2, 1, 3)
        t_neg = t_neg.reshape(B, self.n, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # 计算全局差分系数 λ
        l_exp1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        l_exp2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_val = l_exp1 - l_exp2 + self.lambda_init
        
        # 差分交互：A = Softmax(q @ t_+^T) - λ * Softmax(q @ t_-^T)
        attn_pos = (q * self.scale) @ t_pos.transpose(-2, -1)
        attn_neg = (q * self.scale) @ t_neg.transpose(-2, -1)
        attn = F.softmax(attn_pos, dim=-1) - lambda_val * F.softmax(attn_neg, dim=-1)
        
        # 聚合全局信息
        v_pooled = self.pool(v.reshape(B, H, W, c).permute(0, 3, 1, 2)).reshape(B, c, -1).permute(0, 2, 1)
        v_pooled = v_pooled.reshape(B, self.n, num_heads, head_dim).permute(0, 2, 1, 3)
        
        global_out = attn @ v_pooled # [B, M, N, d]
        global_out = self.subln(global_out) * (1 - self.lambda_init)
        global_out = global_out.transpose(1, 2).reshape(B, N, c)
        global_out = self.proj(global_out)
        
        # --- 步骤 4: 通道门控与最终融合 ---
        f_out = global_out.transpose(1, 2).reshape(B, c, H, W)
        gate_w = self.gate(f_out)
        
        # 动态加权原始投影特征与差分注意力特征
        out = (1 - gate_w) * x_res + gate_w * f_out
        
        return out
```

#### 文件: `LPA.py`

```py
import torch
import torch.nn as nn
#论文：SwinPA-Net: Swin Transformer-Based Multiscale Feature Pyramid Aggregation Network for Medical Image Segmentation
#论文地址：https://ieeexplore.ieee.org/document/9895210

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LPA(nn.Module):
    def __init__(self, in_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4
        x = x3 + x4
        return x


if __name__ == '__main__':

    input = torch.rand(12, 256, 8, 10)
    block = LPA(in_channel=256)
    output = block(input)

    print(input.size())
    print(output.size())
```

#### 文件: `LPA1.py`

```py
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class LPA(nn.Module):
    def __init__(self, in_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        # 将特征图按照高度和宽度分别划分为两半，然后组合成四个象限
        h, w = x.size(2), x.size(3)
        h_half, w_half = h // 2, w // 2
        q1 = x[:, :, :h_half, :w_half]
        q2 = x[:, :, :h_half, w_half:]
        q3 = x[:, :, h_half:, :w_half]
        q4 = x[:, :, h_half:, w_half:]

        # 对每个象限分别应用通道注意力和空间注意力
        q1 = self.sa(self.ca(q1)) * q1
        q2 = self.sa(self.ca(q2)) * q2
        q3 = self.sa(self.ca(q3)) * q3
        q4 = self.sa(self.ca(q4)) * q4

        # 将四个象限还原成完整的特征图
        top = torch.cat([q1, q2], dim=3)
        bottom = torch.cat([q3, q4], dim=3)
        x3 = torch.cat([top, bottom], dim=2)

        # 同时，对整个特征图也做通道和空间注意力
        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4

        # 将局部四象限特征与全局特征进行融合
        out = x3 + x4
        return out

if __name__ == '__main__':
    input = torch.rand(12, 256, 8, 10)
    block = LPA(in_channel=256)
    output = block(input)

    print("Input size:", input.size())
    print("Output size:", output.size())

```

#### 文件: `SMFA1.py`

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Improved SMFA block (new innovation)
# - Adds spatial attention branch
# - Generates both scale (gamma) and shift (beta) modulation maps
# - Fuses local detail and global context via depthwise conv and pooling
# - Applies LayerNorm for stabilized training

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class SMFAv2(nn.Module):
    """
    Spatially-Adaptive Feature Modulation v2
    - Computes per-location gamma and beta for modulation
    - Incorporates local detail and global context
    - Improves on original SMFA by adding shift parameter and attention
    """
    def __init__(self, dim=36, down_scale=8):
        super().__init__()
        self.dim = dim
        self.down_scale = down_scale

        # Projections for query (y) and key (x)
        self.proj_q = nn.Conv2d(dim, dim, 1)
        self.proj_k = nn.Conv2d(dim, dim, 1)

        # Local branch: depthwise conv + pooling
        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pool = nn.AdaptiveAvgPool2d((None, None))

        # Modulation generators: gamma and beta
        self.mod_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1)
        )

        # Detail enhancement
        self.detail = DMlp(dim)

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # Query and Key
        q = self.proj_q(x)  # [B, C, H, W]
        k = self.proj_k(x)  # [B, C, H, W]

        # Local detail
        k_pool = F.adaptive_avg_pool2d(k, (h // self.down_scale, w // self.down_scale))
        k_loc = self.dw_conv(k_pool)
        k_loc = F.interpolate(k_loc, size=(h, w), mode='nearest')

        # Global context: channel-wise statistics
        k_glo = torch.mean(k, dim=(2,3), keepdim=True)

        # Fuse local and global
        fuse = torch.cat([k_loc, k_glo.expand_as(k_loc)], dim=1)  # [B, 2C, H, W]
        gamma_beta = self.mod_conv(fuse)  # [B, 2C, H, W]
        gamma, beta = gamma_beta.chunk(2, dim=1)

        # Apply modulation
        modulated = q * (1 + gamma) + beta

        # Detail enhancement
        detail = self.detail(x)

        out = modulated + detail
        #out = out.permute(0,2,3,1)
        #print(out.shape) 2 36 64 64
        #out = self.norm(out)
        #out = out.permute(0,3,1,2)
        #print(out.shape)
        return out

if __name__ == '__main__':
    inp = torch.randn(2, 36, 64, 64)
    block = SMFAv2(dim=36)
    out = block(inp)
    print('Output shape:', out.shape)

```

#### 文件: `EAFE-M.py`

```py
from __future__ import absolute_import, division, print_function

import time
import json
import datasets.lowcam_dataset as datasets
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac
from models.endodac import DPTHead
import numpy as np
import torch.optim as optim
import networks

from utils.utils import *
from utils.layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import random


class Trim:
    @staticmethod
    def patchify(imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 32
        assert imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    @staticmethod
    def patchify_uncertainty(imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 32
        assert imgs.shape[1] % p == 0

        h = imgs.shape[1] // p
        w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p, w, p))
        x = torch.einsum('nhpwq->nhwpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2))
        return x

    @staticmethod
    def deal_uncertainty_patch(patch):
        uncertainty_patch = patch
        uncertainty_mask = torch.ones((patch.shape[0], patch.shape[1]))

        for i in range(uncertainty_patch.shape[0]):
            k = int(0.75 * len(uncertainty_patch[i]))
            largest_k = torch.topk(uncertainty_patch[i], k)
            k_value = largest_k.values
            k_value = k_value[-1]
            for j in range(uncertainty_patch.shape[1]):

                if uncertainty_patch[i][j] <= k_value:
                    uncertainty_mask[i][j] = 0

            for j in range(uncertainty_patch.shape[1]):
                if uncertainty_mask[i][j] == 0:
                    seed = random.random() > 0.5
                    if seed:
                        uncertainty_mask[i][j] = 1
                elif uncertainty_mask[i][j] == 1:
                    seed = random.random() > 1 - 0.5 * 0.25 / (1 - 0.25)
                    if seed:
                        uncertainty_mask[i][j] = 0

        return uncertainty_mask

    @staticmethod
    def unpatchify(x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 32
        h = 8
        w = 10
        # h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    @staticmethod
    def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device='cuda')  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device='cuda')
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


class Aug:

    def __init__(self):
        pass

    @staticmethod
    def fuse_recon_and_detail(inputs, f_i, models, lam=1.0):
        import torch
        features_recon = models['pretrained_encoder'](inputs["color_aug", f_i, 0])
        recon_outputs = models['pretrained_recon'](features_recon)[('disp', 0)]

        transform_input = [inputs["color_aug", f_i, 0], recon_outputs]
        transform_inputs = models["detail_encoder"](torch.cat(transform_input, 1))
        detail_image = models["detail"](transform_inputs)

        residual_image = detail_image[("transform", 0)] + recon_outputs

        return residual_image


```

#### 文件: `__init__.py`

```py
from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
"""
from .depth_decoder_litemono import DepthDecoderV2
from .pose_decoder import PoseDecoder
from .pose_decode_litemono import PoseDecoderV2
from .pose_cnn import PoseCNN
from .depth_encoder import LiteMono
from .depth_encoder_monovit import mpvit_small
from .depth_decoder_monovit import MonovitDecoder
"""
```

#### 文件: `cednet.py`

```py
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import NORM_LAYERS


BaseBlock = None
FocalBlock = None
act_layer = None
ls_init_value = 1e-6


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first", "channels_first_v2"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

        elif self.data_format == "channels_first_v2":
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class Bottleneck(nn.Module):

    def __init__(self, dim, drop_path=0., norm_cfg=None, **kwargs):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, bias=False),
            build_norm_layer(norm_cfg, dim * 4)[1],
            act_layer(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
        )

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(self.drop_path(x) + input)
        return x


class DBottleneck(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg=None, **kwargs):
        super(DBottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer(),
        )

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, bias=False),
            build_norm_layer(norm_cfg, dim * 4)[1],
            act_layer(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
        )

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x) + x
        x = self.dwconv1(x) + x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(self.drop_path(x) + input)
        return x


class NeXtBlock(nn.Module):

    def __init__(self, dim, drop_path=0., norm_cfg=None, **kwargs):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = build_norm_layer(norm_cfg, dim)[1]
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)    # input (N, C, *)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class DNeXtBlock(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg=None, **kwargs):
        super().__init__()

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv1(x) + x
        x = self.dwconv2(x) + x

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):

    def __init__(self, dim, drop_path=0.0, window_size=7, norm_cfg=None, widx=0, **kwargs):
        super().__init__()

        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.norm_cfg['data_format'] = "channels_last"
        num_heads = dim // 32

        self.dim = dim
        self.window_size = window_size
        self.shift_size = [0, window_size // 2][widx % 2]

        self.norm1 = build_norm_layer(self.norm_cfg, dim)[1]
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads)

        self.norm2 = build_norm_layer(self.norm_cfg, dim)[1]
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_mask = None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # [N, C, H, W] -> [N, H, W, C]

        shortcut = x
        x = self.norm1(x)
        x = self.window_attention(x)
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(x)

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        return x

    def window_attention(self, x):
        _, H, W, C = x.shape

        # # pad feature maps to multiples of window size
        # pad_l = pad_t = 0
        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))    # [B, Hp, Wp, C]
        # _, Hp, Wp, _ = x.shape

        # cyclic shift
        attn_mask = None
        if self.shift_size > 0 and H > self.window_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.attn_mask is None:
                self.attn_mask = self.calculate_attn_mask(H, W).to(x.device)
            attn_mask = self.attn_mask

        # partition windows
        x_windows = window_partition(x, self.window_size)  # [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # if pad_r > 0 or pad_b > 0:
        #     x = x[:, :H, :W, :].contiguous()

        return x

    def calculate_attn_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class DSwinBlock(SwinBlock):

    def __init__(self, dim, drop_path=0.0, dilation=3, window_size=7, norm_cfg=None, widx=0):
        super().__init__(dim, drop_path, window_size, norm_cfg, widx)

        self.norm_act1 = nn.Sequential(
            build_norm_layer(self.norm_cfg, dim)[1],
            act_layer())

        self.dwconv2 = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim)

        self.norm_act2 = nn.Sequential(
            build_norm_layer(self.norm_cfg, dim)[1],
            act_layer())

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # [N, C, H, W] -> [N, H, W, C]

        shortcut = x
        x = self.norm1(x)
        x = self.norm_act1(self.window_attention(x)) + x
        x = self.norm_act2(self.dwconv2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)) + x
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(x)

        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        return x


class Encoder(nn.Module):

    def __init__(self, dims=[192, 352, 512], blocks=[1, 1, 1], dp_rates=0., norm_cfg=None):
        super().__init__()

        assert isinstance(dp_rates, list)
        cum_sum = np.array([0] + blocks[:-1]).cumsum()

        self.encoder = nn.ModuleList([
            nn.Sequential(*[BaseBlock(dims[0], dp_rates[_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[0])]),
            nn.Sequential(*[BaseBlock(dims[1], dp_rates[cum_sum[1]+_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[1])]),
            nn.Sequential(*[FocalBlock(dims[2], dp_rates[cum_sum[2]+_], dilation=3, norm_cfg=norm_cfg, widx=_) for _ in range(blocks[2])]),
        ])

        self.encoder_downsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2), build_norm_layer(norm_cfg, dims[1])[1]),
            nn.Sequential(nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2), build_norm_layer(norm_cfg, dims[2])[1]),
        ])

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        c3 = self.encoder[0](x)
        c4 = self.encoder[1](self.encoder_downsample[0](c3))
        c5 = self.encoder[2](self.encoder_downsample[1](c4))
        return c3, c4, c5


class EncoderFPN(Encoder):

    def forward(self, x):
        if isinstance(x, tuple):
            c3 = self.encoder[0](x[0])
            c4 = self.encoder[1](x[1] + self.encoder_downsample[0](c3))
            c5 = self.encoder[2](x[2] + self.encoder_downsample[1](c4))
        else:
            c3 = self.encoder[0](x)
            c4 = self.encoder[1](self.encoder_downsample[0](c3))
            c5 = self.encoder[2](self.encoder_downsample[1](c4))
        return c3, c4, c5


class Decoder(nn.Module):

    def __init__(self, dims=[192, 352, 512], norm_cfg=None, **kwargs):
        super().__init__()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

    def forward(self, x):
        c3, c4, c5 = x
        c4 = self.decoder_norm[0](c4 + self.decoder_upsample[0](c5))
        c3 = self.decoder_norm[1](c3 + self.decoder_upsample[1](c4))
        return c3, c4, c5


class DecoderFPN(nn.Module):

    def __init__(self, dims=[192, 352, 512], norm_cfg=None, **kwargs):
        super().__init__()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.decoder_conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, dims[0])[1]),
            nn.Sequential(nn.Conv2d(dims[1], dims[1], kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, dims[1])[1]),
            nn.Sequential(nn.Conv2d(dims[2], dims[2], kernel_size=3, stride=1, padding=1), build_norm_layer(norm_cfg, dims[2])[1]),
        )

    def forward(self, x):
        c3, c4, c5 = x
        c4 = self.decoder_norm[0](c4 + self.decoder_upsample[0](c5))
        c3 = self.decoder_norm[1](c3 + self.decoder_upsample[1](c4))

        c3 = self.decoder_conv[0](c3)
        c4 = self.decoder_conv[1](c4)
        c5 = self.decoder_conv[2](c5)
        return c3, c4, c5


class DecoderUNet(nn.Module):

    def __init__(self, dims=[192, 352, 512], blocks=[1, 1], dp_rates=0., norm_cfg=None):
        super().__init__()

        assert isinstance(dp_rates, list)
        cum_sum = np.array([0] + blocks[:-1]).cumsum()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.decoder = nn.Sequential(
            nn.Sequential(*[BaseBlock(dims[1], dp_rates[_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[0])]),
            nn.Sequential(*[BaseBlock(dims[0], dp_rates[cum_sum[0]+_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[1])]),
        )

    def forward(self, x):
        c3, c4, c5 = x
        c4 = self.decoder[0](self.decoder_norm[0](c4 + self.decoder_upsample[0](c5)))
        c3 = self.decoder[1](self.decoder_norm[1](c3 + self.decoder_upsample[1](c4)))
        return c3, c4, c5


class DecoderHourglass(nn.Module):

    def __init__(self, dims=[192, 352, 512], blocks=[1, 1], dp_rates=0., norm_cfg=None):
        super().__init__()

        assert isinstance(dp_rates, list)
        cum_sum = np.array([0] + blocks[:-1]).cumsum()

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[2], dims[1], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)),
            nn.Sequential(nn.Conv2d(dims[1], dims[0], 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        ])

        self.decoder_norm = nn.Sequential(
            build_norm_layer(norm_cfg, dims[1])[1],
            build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.lateral = nn.Sequential(
            BaseBlock(dims[1], norm_cfg=norm_cfg, widx=0),
            BaseBlock(dims[0], norm_cfg=norm_cfg, widx=0),
        )

        self.decoder = nn.Sequential(
            nn.Sequential(*[BaseBlock(dims[1], dp_rates[_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[0])]),
            nn.Sequential(*[BaseBlock(dims[0], dp_rates[cum_sum[0]+_], norm_cfg=norm_cfg, widx=_) for _ in range(blocks[1])]),
        )

    def forward(self, x):
        c3, c4, c5 = x
        c4 = self.decoder[0](self.decoder_norm[0](self.lateral[0](c4) + self.decoder_upsample[0](c5)))
        c3 = self.decoder[1](self.decoder_norm[1](self.lateral[1](c3) + self.decoder_upsample[1](c4)))
        return c3, c4, c5


class CEDNet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 num_stages=3,
                 encoder_type='Encoder',
                 decoder_type='Decoder',
                 base_block='NeXtBlock',
                 focal_block='DNeXtBlock',
                 p2_dim=96,
                 p2_block=3,
                 dims=[192, 352, 512],
                 blocks=[2, 4, 2],
                 layer_scale_init_value=1e-6,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN', eps=1e-6, data_format="channels_first"),
                 act_type='gelu',
                 num_classes=1000,
                 head_init_scale=1.0,
                 **kwargs):
        super().__init__()

        NORM_LAYERS.register_module('LN', force=True, module=LayerNorm)
        global BaseBlock, FocalBlock, act_layer, ls_init_value
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}.get(act_type, None)
        BaseBlock = globals().get(base_block, None)
        FocalBlock = globals().get(focal_block, None)
        ls_init_value = layer_scale_init_value
        self.final_convs = nn.ModuleList([
            nn.Conv2d(2048, 1, kernel_size=1),  # 对应 (8,1,64,80)
            nn.Conv2d(2048, 1, kernel_size=1),  # 对应 (8,1,128,160)
            nn.Conv2d(2048, 1, kernel_size=1)  # 对应 (8,1,256,320)
        ])
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, 2, 1),
            build_norm_layer(norm_cfg, 32)[1],
            act_layer(),
            nn.Conv2d(32, p2_dim, 3, 2, 1),
            build_norm_layer(norm_cfg, p2_dim)[1],
            act_layer(),
        )

        if isinstance(blocks[0], int):
            blocks = [blocks for _ in range(num_stages)]

        max_num_blocks = p2_block + np.array(blocks).sum()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, max_num_blocks)]

        self.stages = nn.ModuleList()

        # p2 stage
        p2_stage = [BaseBlock(p2_dim, drop_path=dp_rates[_], norm_cfg=norm_cfg, widx=_) for _ in range(p2_block)]
        p2_stage.append(nn.Sequential(
            nn.Conv2d(p2_dim, dims[0], kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, dims[0])[1]))
        self.stages.append(nn.Sequential(*p2_stage))

        # CEDNet stages
        for sidx in range(num_stages):
            sta_idx = p2_block + np.array(blocks[:sidx], dtype=np.int32).sum()
            end_dix = p2_block + np.array(blocks[:sidx+1], dtype=np.int32).sum()

            rates = dp_rates[sta_idx:end_dix]
            encoder_blocks = [blocks[sidx][0] // 2, blocks[sidx][1] // 2, blocks[sidx][2]]
            decoder_blocks = [blocks[sidx][1] - blocks[sidx][1] // 2, blocks[sidx][0] - blocks[sidx][0] // 2]
            if decoder_type in ['Decoder', 'DecoderFPN']:
                encoder_blocks = blocks[sidx]

            stage = [globals()[encoder_type](
                dims=dims, blocks=encoder_blocks, dp_rates=rates[:sum(encoder_blocks)], norm_cfg=norm_cfg)]
            if sidx < num_stages - 1:
                stage.append(globals()[decoder_type](
                    dims=dims, blocks=decoder_blocks, dp_rates=rates[sum(encoder_blocks):], norm_cfg=norm_cfg))
            self.stages.append(nn.Sequential(*stage))

        self.final_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _ in range(num_stages):
            self.final_layers.append(
                nn.Sequential(
                    nn.Conv2d(dims[-1], 2048, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, 2048)[1],
                    act_layer(),
                )
            )
            self.norms.append(nn.LayerNorm(1, eps=1e-6))
            self.heads.append(nn.Linear(2048, num_classes))

        self.apply(self._init_weights)
        for head in self.heads:
            head.weight.data.mul_(head_init_scale)
            head.bias.data.mul_(head_init_scale)

        NORM_LAYERS.register_module('LN', force=True, module=nn.LayerNorm)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_training(self, x):
        x = self.stem(x)
        x = self.stages[0](x)

        stage_feats = []
        for stage in self.stages[1:]:

            x = stage(x)

            stage_feats.append(x)


        outs = []
        outs1 = {}
        for i, feats in enumerate(stage_feats):
            out = self.final_layers[i](feats[-1])
            #out = self.heads[i](self.norms[i](out.mean([-2, -1])))
            outs.append(out)
        outs1[("disp",0)] = F.interpolate(self.final_convs[2](outs[0]), size=(256, 320), mode='bilinear', align_corners=False)
        outs1[("disp",1)] = F.interpolate(self.final_convs[1](outs[1]), size=(128, 160), mode='bilinear', align_corners=False)
        outs1[("disp",2)] = F.interpolate(self.final_convs[0](outs[2]), size=(64, 80), mode='bilinear', align_corners=False)
        return outs1

    def forward_inference(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        out = self.final_layers[-1](x[-1])
        out = self.heads[-1](self.norms[-1](out.mean([-2, -1])))
        return out

    def forward(self, x):
        return self.forward_training(x)# if self.training else self.forward_inference(x)


@register_model
def CEDNet_next_tiny(**kwargs): # drop_path_rate = 0.1
    model = CEDNet(num_stages=3, p2_dim=96, p2_block=3, dims=[192, 352, 512], blocks=[2, 4, 2], **kwargs)
    return model


@register_model
def CEDNet_next_small(**kwargs): # drop_path_rate = 0.4
    model = CEDNet(num_stages=4, p2_dim=96, p2_block=3, dims=[192, 352, 512], blocks=[2, 7, 2], **kwargs)
    return model


@register_model
def CEDNet_next_base(**kwargs): # drop_path_rate = 0.5
    model = CEDNet(num_stages=4, p2_dim=128, p2_block=3, dims=[256, 448, 704], blocks=[2, 7, 2], **kwargs)
    return model


@register_model
def CEDNet_swin_tiny(**kwargs): # drop_path_rate = 0.1
    model = CEDNet(
        base_block='SwinBlock', focal_block='DSwinBlock',
        num_stages=3, p2_dim=96, p2_block=3, dims=[160, 288, 416], blocks=[2, 4, 2],
        **kwargs)
    return model


@register_model
def CEDNet_res50(**kwargs): # drop_path_rate = 0.
    model = CEDNet(
        base_block='Bottleneck', focal_block='DBottleneck',
        num_stages=3, p2_dim=64, p2_block=3, dims=[128, 256, 384], blocks=[1, 4, 2],
        norm_cfg=dict(type='BN', requires_grad=True), act_type='relu', **kwargs)
    return model


@register_model
def CEDNet_next_tiny_fpn(**kwargs):
    model = CEDNet(
        encoder_type='EncoderFPN',
        decoder_type='DecoderFPN',
        num_stages=3, p2_dim=96, p2_block=3, dims=[160, 304, 448], blocks=[2, 4, 2], **kwargs)
    return model


@register_model
def CEDNet_next_tiny_unet(**kwargs):
    model = CEDNet(
        decoder_type='DecoderUNet',
        num_stages=3, p2_dim=96, p2_block=3, dims=[192, 352, 512], blocks=[2, 4, 2], **kwargs)
    return model


@register_model
def CEDNet_next_tiny_hourglass(**kwargs):
    model = CEDNet(
        decoder_type='DecoderHourglass',
        num_stages=3, p2_dim=96, p2_block=3, dims=[160, 320, 512], blocks=[2, 4, 2], **kwargs)
    return model



if __name__ == "__main__":

    import torch

    # 创建模型
    #model = CEDNet_next_tiny()
    model = CEDNet_next_tiny_fpn()


    input_tensor = torch.randn(8, 3, 256,320)

# 运行前向传播
    with torch.no_grad():
        outputs = model(input_tensor)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)




```

#### 文件: `common.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

```

#### 文件: `contmix.py`

```py
'''
This is a plug-and-play implementation of ContMix block in the paper:
https://arxiv.org/abs/2502.20087
'''
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from timm.models.layers import DropPath, to_2tuple
from torch.utils.checkpoint import checkpoint

try:
    from natten.functional import na2d_av
    has_natten = True
except:
    has_natten = False
    warnings.warn("The efficiency may be reduced since 'natten' is not installed."
                  " It is recommended to install natten for better performance.")


def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl=True):

    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):

        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])

        return x


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()



class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))


    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x



class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class ResDWConv(nn.Conv2d):
    '''
    Depthwise conv with residual connection
    '''
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        x = x + super().forward(x)
        return x


class ContMixBlock(nn.Module):
    '''
    A plug-and-play implementation of ContMix module with FFN layer
    Paper: https://arxiv.org/abs/2502.20087
    '''
    def __init__(self,
                 dim=64,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=4,
                 res_scale=False,
                 ls_init_value=None,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 **kwargs):

        super().__init__()
        '''
        Args:
        kernel_size: kernel size of the main ContMix branch, default is 7
        smk_size: kernel size of the secondary ContMix branch, default is 5
        num_heads: number of dynamic kernel heads, default is 2
        mlp_ratio: ratio of mlp hidden dim to embedding dim, default is 4
        res_scale: whether to use residual layer scale, default is False
        ls_init_value: layer scale init value, default is None
        drop_path: drop path rate, default is 0
        norm_layer: normalization layer, default is LayerNorm2d
        use_gemm: whether to use iGEMM implementation for large kernel conv, default is False
        deploy: whether to use deploy mode, default is False
        use_checkpoint: whether to use grad checkpointing, default is False
        **kwargs: other arguments
        '''
        mlp_dim = int(dim*mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.dwconv1 = ResDWConv(dim, kernel_size=3)
        self.norm1 = norm_layer(dim)

        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )
        self.weight_value = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        )
        self.se_layer = SEModule(dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        self.dwconv2 = ResDWConv(dim, kernel_size=3)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
        )

        self.ls1 = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.get_rpb()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.trunc_normal_(self.rpb1, std=0.02)
        nn.init.trunc_normal_(self.rpb2, std=0.02)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        """
        RPB implementation directly borrowed from https://tinyurl.com/mrbub4t3
        """
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*kernel_size-1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size**2))
        return attn + rpb

    def reparm(self):
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()

    def _forward_inner(self, x):
        input_resolution = x.shape[2:]
        B, C, H, W = x.shape

        x = self.dwconv1(x)
        identity = x
        x = self.norm1(x)
        gate = self.gate(x)
        lepe = self.lepe(x)

        is_pad = False
        if min(H, W) < self.kernel_size:
            is_pad = True
            if H < W:
                size = (self.kernel_size, int(self.kernel_size / H * W))
            else:
                size = (int(self.kernel_size / W * H), self.kernel_size)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            H, W = size

        query = self.weight_query(x) * self.scale
        key = self.weight_key(x)
        value = self.weight_value(x)

        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)

        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size**2, self.kernel_size**2], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(value, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)

        if has_natten:
            x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
            x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)
        else:
            pad1 = self.smk_size // 2
            pad2 = self.kernel_size // 2
            H_o1 = H - 2 * pad1
            W_o1 = W - 2 * pad1
            H_o2 = H - 2 * pad2
            W_o2 = W - 2 * pad2

            v1 = rearrange(value[0], 'b g h w c -> b (g c) h w')
            v2 = rearrange(value[1], 'b g h w c -> b (g c) h w')

            v1 = F.unfold(v1, kernel_size=self.smk_size).reshape(B, -1, H_o1, W_o1)
            v2 = F.unfold(v2, kernel_size=self.kernel_size).reshape(B, -1, H_o2, W_o2)

            v1 = F.pad(v1, (pad1, pad1, pad1, pad1), mode='replicate')
            v2 = F.pad(v2, (pad2, pad2, pad2, pad2), mode='replicate')

            v1 = rearrange(v1, 'b (g c k) h w -> b g c h w k', g=self.num_heads, k=self.smk_size**2, h=H, w=W)
            v2 = rearrange(v2, 'b (g c k) h w -> b g c h w k', g=self.num_heads, k=self.kernel_size**2, h=H, w=W)

            x1 = einsum(attn1, v1, 'b g h w k, b g c h w k -> b g h w c')
            x2 = einsum(attn2, v2, 'b g h w k, b g c h w k -> b g h w c')

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)

        if is_pad:
            x = F.adaptive_avg_pool2d(x, input_resolution)

        x = self.fusion_proj(x)

        x = x + lepe
        x = self.se_layer(x)

        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))

        x = self.dwconv2(x)

        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_inner, x, use_reentrant=False)
        else:
            x = self._forward_inner(x)
        return x


if __name__ == '__main__':

    from timm.utils import random_seed
    random_seed(6)

    x = torch.randn(1, 64, 32, 32).cuda()
    model = ContMixBlock(dim=64,
                         num_heads=2,
                         kernel_size=13,
                         smk_size=5,
                         mlp_ratio=4,
                         res_scale=True,
                         ls_init_value=1,
                         drop_path=0,
                         norm_layer=LayerNorm2d,
                         use_gemm=True,
                         deploy=False,
                         use_checkpoint=False)
    print(model)
    model.cuda()
    model.eval()
    y = model(x)
    print(y.shape)

    # Reparametrize model, more details can be found at:
    # https://github.com/AILab-CVC/UniRepLKNet/tree/main
    model.reparm()
    z = model(x)

    # Showing difference between original and reparametrized model
    print((z - y).abs().sum() / y.abs().sum())
```

#### 文件: `custom_modules.py`

```py
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
```

#### 文件: `depth_decoder.py`

```py
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
```

#### 文件: `depth_decoder_litemono.py`

```py
from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from .layers import *
from timm.models.layers import trunc_normal_


class DepthDecoderV2(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # print(i, num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs


```

#### 文件: `depth_decoder_monovit.py`

```py
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .hr_layers import *


class MonovitDecoder(nn.Module):
    def __init__(self, ch_enc = [64,128,216,288,288], scales=range(4),num_ch_enc = [ 64, 64, 128, 256, 512 ], num_output_channels=1):
        super(MonovitDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        
        # decoder
        self.convs = nn.ModuleDict()
        
        # feature fusion
        self.convs["f4"] = Attention_Module(self.ch_enc[4]  , num_ch_enc[4])
        self.convs["f3"] = Attention_Module(self.ch_enc[3]  , num_ch_enc[3])
        self.convs["f2"] = Attention_Module(self.ch_enc[2]  , num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.ch_enc[1]  , num_ch_enc[1])
        


        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                        self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        for i in range(4):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
                

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        feat={}
        feat[4] = self.convs["f4"](input_features[4])
        feat[3] = self.convs["f3"](input_features[3])
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = input_features[0]
        
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[("disp", 0)] = self.sigmoid(self.convs["dispconv0"](x))
        outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](features["X_04"]))
        outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv2"](features["X_13"]))
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features["X_22"]))
        return outputs
        
```

#### 文件: `depth_encoder.py`

```py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x)
        x = self.bn1(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class LGFI(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)

        return x


class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x


class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()

        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 10]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]
            else:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
                

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 7]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 7]
            self.dims = [32, 64, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-8m':
            self.num_ch_enc = np.array([64, 128, 224])
            self.depth = [4, 4, 10]
            self.dims = [64, 128, 224]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.stem2 = nn.Sequential(
            Conv(self.dims[0]+3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i]*2+3, self.dims[i+1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        x = (x - 0.45) / 0.225

        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))
        tmp_x.append(x)

        for s in range(len(self.stages[0])-1):
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x)
        tmp_x.append(x)
        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)

            tmp_x = [x]
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
            tmp_x.append(x)

            features.append(x)

        return features

    def forward(self, x):
        x = self.forward_features(x)

        return x

```

#### 文件: `depth_encoder_monovit.py`

```py
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# CoaT: https://github.com/mlpc-ucsd/CoaT
# --------------------------------------------------------------------------------


import numpy as np
import math

import torch

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_

from einops import rearrange
from functools import partial
from torch import nn, einsum
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.runner import load_checkpoint,load_state_dict
from mmcv.cnn import build_norm_layer

from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

__all__ = [
    "mpvit_tiny",
    "mpvit_xsmall",
    "mpvit_small",
    "mpvit_base",
]

def _cfg_mpvit(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a. MLP) class."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act_layer=None,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()
        # self.add_module('c', torch.nn.Conv2d(
        #     a, b, ks, stride, pad, dilation, groups, bias=False))
        self.conv = torch.nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False
        )
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):
    """
    Depthwise Separable Conv
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """
    Depthwise Convolutional Patch Embedding layer
    Image to Patch Embedding
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=768,
        patch_size=16,
        stride=1,
        pad=0,
        act_layer=nn.Hardswish,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        # TODO : confirm whether act_layer is effective or not
        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    def __init__(self, embed_dim, num_path=4, isPool=False, norm_cfg=dict(type="BN")):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList(
            [
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=2 if isPool and idx == 0 else 1,
                    pad=1,
                    norm_cfg=norm_cfg,
                )
                for idx in range(num_path)
            ]
        )

        # scale

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
                )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = (
            x.transpose(1, 2).reshape(B, N, C).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MHCABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3,
        drop_path=0.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        shared_cpe=None,
        shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        # x.shape = [B, N, C]

        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x


class MHCAEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        drop_path_list=[],
        qk_scale=None,
        crpe_window={3: 2, 5: 3, 7: 3},
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window=crpe_window)
        self.MHCA_layers = nn.ModuleList(
            [
                MHCABlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_list[idx],
                    qk_scale=qk_scale,
                    shared_cpe=self.cpe,
                    shared_crpe=self.crpe,
                )
                for idx in range(self.num_layers)
            ]
        )

    def forward(self, x, size):
        H, W = size
        B = x.shape[0]
        # x' shape : [B, N, C]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Hardswish,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(
            in_features, hidden_features, act_layer=act_layer, norm_cfg=norm_cfg
        )
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        # self.norm = norm_layer(hidden_features)
        self.norm = build_norm_layer(norm_cfg, hidden_features)[1]
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MHCA_stage(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_embed_dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        num_path=4,
        norm_cfg=dict(type="BN"),
        drop_path_list=[],
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList(
            [
                MHCAEncoder(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                )
                for _ in range(num_path)
            ]
        )

        self.InvRes = ResBlock(
            in_features=embed_dim, out_features=embed_dim, norm_cfg=norm_cfg
        )
        self.aggregate = Conv2d_BN(
            embed_dim * (num_path + 1),
            out_embed_dim,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, inputs):
        att_outputs = [self.InvRes(inputs[0])]
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            att_outputs.append(encoder(x, size=(H, W)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)

        return out,att_outputs


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """
    Generate drop path rate list following linear decay rule
    """
    dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur : cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


@BACKBONES.register_module()
class MPViT(nn.Module):
    """Multi-Path ViT class."""

    def __init__(
        self,
        num_classes=80,
        in_chans=3,
        num_stages=4,
        num_layers=[1, 1, 1, 1],
        mlp_ratios=[8, 8, 4, 4],
        num_path=[4, 4, 4, 4],
        embed_dims=[64, 128, 256, 512],
        num_heads=[8, 8, 8, 8],
        drop_path_rate=0.2,
        norm_cfg=dict(type="BN"),
        norm_eval=False,
        pretrained=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages
        self.conv_norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                norm_cfg=self.conv_norm_cfg,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=1,
                pad=1,
                act_layer=nn.Hardswish,
                norm_cfg=self.conv_norm_cfg,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList(
            [
                Patch_Embed_stage(
                    embed_dims[idx],
                    num_path=num_path[idx],
                    isPool= True,
                    norm_cfg=self.conv_norm_cfg,
                )
                for idx in range(self.num_stages)
            ]
        )

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList(
            [
                MHCA_stage(
                    embed_dims[idx],
                    embed_dims[idx + 1]
                    if not (idx + 1) == self.num_stages
                    else embed_dims[idx],
                    num_layers[idx],
                    num_heads[idx],
                    mlp_ratios[idx],
                    num_path[idx],
                    norm_cfg=self.conv_norm_cfg,
                    drop_path_list=dpr[idx],
                )
                for idx in range(self.num_stages)
            ]
        )

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_features(self, x):

        # x's shape : [B, C, H, W]
        outs = []
        x = self.stem(x)  # Shape : [B, C, H/4, W/4]
        outs.append(x)
        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)
            #outs.append(att_inputs)
            x,ff = self.mhca_stages[idx](att_inputs)
            outs.append(x)


        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(MPViT, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def mpvit_tiny(**kwargs):
    """mpvit_tiny :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 96, 176, 216]
    - MLP_ratio : 2
    Number of params: 5843736
    FLOPs : 1654163812
    Activations : 16641952
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 96, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_xsmall(**kwargs):
    """mpvit_xsmall :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 128, 192, 256]
    - MLP_ratio : 4
    Number of params : 10573448
    FLOPs : 2971396560
    Activations : 21983464
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 128, 192, 256],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    checkpoint = torch.load('./ckpt/mpvit_xsmall.pth', map_location=lambda storage, loc: storage)['model']
    logger = get_root_logger()
    load_state_dict(model, checkpoint, strict=False, logger=logger)
    del checkpoint
    del logger
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_small(**kwargs):
    """mpvit_small :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 6, 3]
    - #channels : [64, 128, 216, 288]
    - MLP_ratio : 4
    Number of params : 22892400
    FLOPs : 4799650824
    Activations : 30601880
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    checkpoint = torch.load('./pretrain/mpvit_small.pth', map_location=lambda storage, loc: storage)['model']
    logger = get_root_logger()
    load_state_dict(model, checkpoint, strict=False, logger=logger)
    del checkpoint
    del logger
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_base(**kwargs):
    """mpvit_base :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 8, 3]
    - #channels : [128, 224, 368, 480]
    - MLP_ratio : 4
    Number of params: 74845976
    FLOPs : 16445326240
    Activations : 60204392
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 8, 3],
        embed_dims=[128, 224, 368, 480],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model
```

#### 文件: `ghostnetv3.py`

```py
# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, List, Tuple
from timm.models.registry import register_model

#__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
    
    
def gcd(a,b):
    if a<b:
        a,b=b,a
    while(a%b != 0):
        c = a%b
        a=b
        b=c
    return b
    
def MyNorm(dim):
    return nn.GroupNorm(1, dim)  

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode=None,args=None):
        super(GhostModule, self).__init__()
        #self.args=args
        self.mode = mode
        self.gate_loc = 'before'
        
        self.inter_mode = 'nearest'
        self.scale = 1.0
        
        self.infer_mode = False
        self.num_conv_branches = 3
        self.dconv_scale = True
        self.gate_fn = nn.Sigmoid()

        # if args.gate_fn=='hard_sigmoid':
        #     self.gate_fn=hard_sigmoid
        # elif args.gate_fn=='sigmoid': 
        #     self.gate_fn=nn.Sigmoid()
        # elif args.gate_fn=='relu': 
        #     self.gate_fn=nn.ReLU()
        # elif args.gate_fn=='clip': 
        #     self.gate_fn=myclip 
        # elif args.gate_fn=='tanh': 
        #     self.gate_fn=nn.Tanh()

        if self.mode in ['ori']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            if self.infer_mode:  
                self.primary_conv = nn.Sequential(  
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None


                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels, bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size
     
        elif self.mode in ['ori_shortcut_mul_conv15']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            )
            if self.infer_mode:
                self.primary_conv = nn.Sequential(  
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                ) 
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None


                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels, bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size

      
    def forward(self, x):
        if self.mode in ['ori']:
            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1,x2], dim=1)
            return out

        elif self.mode in ['ori_shortcut_mul_conv15']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))
            
            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1,x2], dim=1)

            if self.gate_loc=='before':
                return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res/self.scale),size=out.shape[-2:],mode=self.inter_mode) # 'nearest'
#                 return out*F.interpolate(self.gate_fn(res/self.scale),size=out.shape[-1].item(),mode=self.inter_mode) # 'nearest'
            else:
                return out[:,:self.oup,:,:]*self.gate_fn(F.interpolate(res,size=out.shape[-2:],mode=self.inter_mode))  
#                 return out*self.gate_fn(F.interpolate(res,size=out.shape[-1],mode=self.inter_mode))  


    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.infer_mode:
            return
        primary_kernel, primary_bias = self._get_kernel_bias_primary()
        self.primary_conv = nn.Conv2d(in_channels=self.primary_rpr_conv[0].conv.in_channels,
                                      out_channels=self.primary_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.primary_rpr_conv[0].conv.kernel_size,
                                      stride=self.primary_rpr_conv[0].conv.stride,
                                      padding=self.primary_rpr_conv[0].conv.padding,
                                      dilation=self.primary_rpr_conv[0].conv.dilation,
                                      groups=self.primary_rpr_conv[0].conv.groups,
                                      bias=True)
        self.primary_conv.weight.data = primary_kernel
        self.primary_conv.bias.data = primary_bias
        self.primary_conv = nn.Sequential(
            self.primary_conv, 
            self.primary_activation if self.primary_activation is not None else nn.Sequential()
        )

        cheap_kernel, cheap_bias = self._get_kernel_bias_cheap()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_rpr_conv[0].conv.in_channels,
                                      out_channels=self.cheap_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.cheap_rpr_conv[0].conv.kernel_size,
                                      stride=self.cheap_rpr_conv[0].conv.stride,
                                      padding=self.cheap_rpr_conv[0].conv.padding,
                                      dilation=self.cheap_rpr_conv[0].conv.dilation,
                                      groups=self.cheap_rpr_conv[0].conv.groups,
                                      bias=True)
        self.cheap_operation.weight.data = cheap_kernel
        self.cheap_operation.bias.data = cheap_bias

        self.cheap_operation = nn.Sequential(
            self.cheap_operation, 
            self.cheap_activation if self.cheap_activation is not None else nn.Sequential()
        )

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'primary_rpr_conv'):
            self.__delattr__('primary_rpr_conv')
        if hasattr(self, 'primary_rpr_scale'):
            self.__delattr__('primary_rpr_scale')
        if hasattr(self, 'primary_rpr_skip'):
            self.__delattr__('primary_rpr_skip')

        if hasattr(self, 'cheap_rpr_conv'):
            self.__delattr__('cheap_rpr_conv')
        if hasattr(self, 'cheap_rpr_scale'):
            self.__delattr__('cheap_rpr_scale')
        if hasattr(self, 'cheap_rpr_skip'):
            self.__delattr__('cheap_rpr_skip')

        self.infer_mode = True

    def _get_kernel_bias_primary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.primary_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.primary_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.primary_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.primary_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.primary_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final
    
    def _get_kernel_bias_cheap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.cheap_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.cheap_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.cheap_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.cheap_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.cheap_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list


class GhostBottleneck(nn.Module): 
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,layer_id=None,args=None):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        self.num_conv_branches = 3
        self.infer_mode = False
        self.dconv_scale = True

        # Point-wise expansion
        if layer_id<=1:
            self.ghost1 = GhostModule(in_chs, mid_chs, relu=True,mode='ori',args=args)
        else:
            self.ghost1 = GhostModule(in_chs, mid_chs, relu=True,mode='ori_shortcut_mul_conv15',args=args) ####这里是扩张 mid_chs远大于in_chs

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                 padding=(dw_kernel_size-1)//2,
                                 groups=mid_chs, bias=False)
                self.bn_dw = nn.BatchNorm2d(mid_chs)
            else:
                self.dw_rpr_skip = nn.BatchNorm2d(mid_chs) if stride == 1 else None
                dw_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    dw_rpr_conv.append(self._conv_bn(mid_chs, mid_chs, dw_kernel_size, stride, (dw_kernel_size-1)//2, groups=mid_chs, bias=False))
                self.dw_rpr_conv = nn.ModuleList(dw_rpr_conv)
                # Re-parameterizable scale branch
                self.dw_rpr_scale = None
                if dw_kernel_size > 1:
                    self.dw_rpr_scale = self._conv_bn(mid_chs, mid_chs, 1, 2, 0, groups=mid_chs, bias=False)
                self.kernel_size = dw_kernel_size
                self.in_channels = mid_chs

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        if layer_id<=1:
            self.ghost2 = GhostModule(mid_chs, out_chs, relu=False,mode='ori',args=args)
        else:
            self.ghost2 = GhostModule(mid_chs, out_chs, relu=False,mode='ori',args=args)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                x = self.conv_dw(x)
                x = self.bn_dw(x)
            else:
                dw_identity_out = 0
                if self.dw_rpr_skip is not None:
                    dw_identity_out = self.dw_rpr_skip(x)
                dw_scale_out = 0
                if self.dw_rpr_scale is not None and self.dconv_scale:
                    dw_scale_out = self.dw_rpr_scale(x)
                x1 = dw_scale_out + dw_identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.dw_rpr_conv[ix](x)
                x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list


    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.infer_mode or self.stride == 1:
            return
        dw_kernel, dw_bias = self._get_kernel_bias_dw()
        self.conv_dw = nn.Conv2d(in_channels=self.dw_rpr_conv[0].conv.in_channels,
                                      out_channels=self.dw_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.dw_rpr_conv[0].conv.kernel_size,
                                      stride=self.dw_rpr_conv[0].conv.stride,
                                      padding=self.dw_rpr_conv[0].conv.padding,
                                      dilation=self.dw_rpr_conv[0].conv.dilation,
                                      groups=self.dw_rpr_conv[0].conv.groups,
                                      bias=True)
        self.conv_dw.weight.data = dw_kernel
        self.conv_dw.bias.data = dw_bias
        self.bn_dw = nn.Identity()

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'dw_rpr_conv'):
            self.__delattr__('dw_rpr_conv')
        if hasattr(self, 'dw_rpr_scale'):
            self.__delattr__('dw_rpr_scale')
        if hasattr(self, 'dw_rpr_skip'):
            self.__delattr__('dw_rpr_skip')

        self.infer_mode = True

    def _get_kernel_bias_dw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.dw_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.dw_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.dw_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.dw_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.dw_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final


    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, block=GhostBottleneck, args=None):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id=0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block==GhostBottleneck:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio,layer_id=layer_id,args=args))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        features = []
        for i,stage in enumerate(self.blocks):
            x = stage(x)
            if i in[0,2,4,6,8]:

                features.append(x)
        #x = self.blocks(x)

        #x = self.global_pool(x)
        #x = self.conv_head(x)
        #x = self.act2(x)
        #x = x.view(x.size(0), -1)
        # if self.dropout > 0.:
            # x = F.dropout(x, p=self.dropout, training=self.training)
        #x = self.classifier(x)
        #x = x.squeeze()
        return features

    def reparameterize(self):
        for _, module in self.named_modules():
            if isinstance(module, GhostModule):
                module.reparameterize()
            if isinstance(module, GhostBottleneck):
                module.reparameterize()

@register_model
def ghostnetv3(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, num_classes=1000, width=kwargs['width'], dropout=0.2)

if __name__=='__main__':
    model = ghostnetv3(width=1.0)
    model.eval()
    print(model)
    input1 = torch.randn(32,3,320,256)
    input2 = torch.randn(8,3,256,320)
    input3 = torch.randn(32,3,224,224)

    with torch.inference_mode():
       # y11 = model(input1)
        y12 = model(input2)

        #y13 = model(input3)
    """
    model.reparameterize()
    print(model)
    with torch.inference_mode():
        y21 = model(input1)
        y22 = model(input2)
        y23 = model(input3)
    print(torch.allclose(y11, y21), torch.norm(y11 - y21))
    print(torch.allclose(y12, y22), torch.norm(y12 - y22))
    print(torch.allclose(y13, y23), torch.norm(y13 - y23))
    """

```

#### 文件: `hr_layers.py`

```py
from __future__ import absolute_import, division, print_function

import numpy as np
import math

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F




def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def visual_feature(features,stage):
    feature_map = features.squeeze(0).cpu()
    n,h,w = feature_map.size()
    print(h,w)
    list_mean = []
    #sum_feature_map = torch.sum(feature_map,0)
    sum_feature_map,_ = torch.max(feature_map,0)
    for i in range(n):
        list_mean.append(torch.mean(feature_map[i]))
        
    sum_mean = sum(list_mean)
    feature_map_weighted = torch.ones([n,h,w])
    for i in range(n):
        feature_map_weighted[i,:,:] = (torch.mean(feature_map[i]) / sum_mean) * feature_map[i,:,:]
    sum_feature_map_weighted = torch.sum(feature_map_weighted,0)
    plt.imshow(sum_feature_map)
    #plt.savefig('feature_viz/{}_stage.png'.format(a))
    plt.savefig('feature_viz/decoder_{}.png'.format(stage))
    plt.imshow(sum_feature_map_weighted)
    #plt.savefig('feature_viz/{}_stage_weighted.png'.format(a))
    plt.savefig('feature_viz/decoder_{}_weighted.png'.format(stage))

def depth_to_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = 1 / depth - min_disp
    return disp / (max_disp - min_disp)

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.atrous_block1  = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.atrous_block6  = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)

        self.conv1x1 = nn.Conv2d(out_channels*4, out_channels, 1, 1)

    def forward(self, features):
        features_1 = self.atrous_block18(features[0])
        features_2 = self.atrous_block12(features[1])
        features_3 = self.atrous_block6(features[2])
        features_4 = self.atrous_block1(features[3])

        output_feature = [features_1, features_2, features_3, features_4]
        output_feature = torch.cat(output_feature, 1)

        return self.conv1x1(output_feature)

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # Prepare Coordinates shape [b,3,h*w]
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        # normalize
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class SE_block(nn.Module):
    def __init__(self, in_channel, visual_weights = False, reduction = 16 ):
        super(SE_block, self).__init__()
        reduction = reduction
        in_channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channel // reduction, in_channel, bias = False)
            )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace = True)
        self.vis = False
    
    def forward(self, in_feature):

        b,c,_,_ = in_feature.size()
        output_weights_avg = self.avg_pool(in_feature).view(b,c)
        output_weights_max = self.max_pool(in_feature).view(b,c)
        output_weights_avg = self.fc(output_weights_avg).view(b,c,1,1)
        output_weights_max = self.fc(output_weights_max).view(b,c,1,1)
        output_weights = output_weights_avg + output_weights_max
        output_weights = self.sigmoid(output_weights)
        return output_weights.expand_as(in_feature) * in_feature

## ChannelAttetion
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Linear(in_planes,in_planes // ratio, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_planes // ratio, in_planes, bias = False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b,c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature

## SpatialAttetion

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, in_feature):
        x = in_feature
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #x = avg_out
        #x = max_out
        x = self.conv1(x)
        return self.sigmoid(x).expand_as(in_feature) * in_feature


#CS means channel-spatial  
class CS_Block(nn.Module):
    def __init__(self, in_channel, reduction = 16 ):
        super(CS_Block, self).__init__()
        
        reduction = reduction
        in_channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channel // reduction, in_channel, bias = False)
            )
        self.sigmoid = nn.Sigmoid()
        ## Spatial_Block
        self.conv = nn.Conv2d(2,1,kernel_size = 1, bias = False)
        #self.conv = nn.Conv2d(1,1,kernel_size = 1, bias = False)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, in_feature):

        b,c,_,_ = in_feature.size()
        
        
        output_weights_avg = self.avg_pool(in_feature).view(b,c)
        output_weights_max = self.max_pool(in_feature).view(b,c)
         
        output_weights_avg = self.fc(output_weights_avg).view(b,c,1,1)
        output_weights_max = self.fc(output_weights_max).view(b,c,1,1)
        
        output_weights = output_weights_avg + output_weights_max
        
        output_weights = self.sigmoid(output_weights)
        out_feature_1 = output_weights.expand_as(in_feature) * in_feature
        
        ## Spatial_Block
        in_feature_avg = torch.mean(out_feature_1,1,True)
        in_feature_max,_ = torch.max(out_feature_1,1,True)
        mixed_feature = torch.cat([in_feature_avg,in_feature_max],1)
        spatial_attention = self.sigmoid(self.conv(mixed_feature))
        out_feature = spatial_attention.expand_as(out_feature_1) * out_feature_1
        #########################
        
        return out_feature
        
class Attention_Module(nn.Module):
    def __init__(self, high_feature_channel, output_channel = None):
        super(Attention_Module, self).__init__()
        in_channel = high_feature_channel 
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.ca = ChannelAttention(channel)
        #self.sa = SpatialAttention()
        #self.cs = CS_Block(channel)
        self.conv_se = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1 )
        self.relu = nn.ReLU(inplace = True)

    def forward(self, high_features):

        features = high_features

        features = self.ca(features)
        #features = self.sa(features)
        #features = self.cs(features)
        
        return self.relu(self.conv_se(features))

class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))
```

#### 文件: `intrinsics_decoder.py`

```py
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
```

#### 文件: `layers.py`

```py
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia as K
import kornia.feature as KF


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def depth_to_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)
    return disp
    


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlockDepth(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDepth, self).__init__()

        self.conv = DepthConv3x3(in_channels, out_channels)
        self.nonlin = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(DepthConv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=3 // 2, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points # (b, 4, h*w)


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, compute_depth=False):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points) # (B, 3, h*w)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        
        if not compute_depth:
            return pix_coords
        else:
            computed_depth = cam_points[:, 2, :].unsqueeze(1).view(self.batch_size, 1, self.height, self.width)
            return pix_coords, computed_depth
    
    

def upsample(x, scale_factor=2, mode="bilinear"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class LoFTR(nn.Module):
    """Layer to compute the correspondences between a pair of images
    """
    def __init__(self, pretrained='indoor'):
        super(LoFTR, self).__init__()
        self.matcher = KF.LoFTR(pretrained=pretrained)

    def forward(self, src0, srcx):
        input_dict = {"image0": K.color.rgb_to_grayscale(src0), # LofTR works on grayscale images only 
                    "image1": K.color.rgb_to_grayscale(srcx)}
        with torch.no_grad():
            correspondences = self.matcher(input_dict)
        return correspondences

def compute_matcher_errors(correspondences, sample, width, height, bs):
    """Computation of matcher error between reprojected points and pseudo labels
    """
    # convert sample to pixel_coords
    raw_pix_coords = sample / 2 + 0.5
    raw_pix_coords[..., 0] *= width - 1
    raw_pix_coords[..., 1] *= height - 1
    loss = 0
    # calculate matcher error for each image in the batch
    for bs_i in range(bs):
        tensor_kp0 = correspondences['keypoints0'][(correspondences['batch_indexes'] == bs_i)].detach()
        tensor_kp1 = correspondences['keypoints1'][(correspondences['batch_indexes'] == bs_i)].detach()

        selected_idx = (tensor_kp0[:, 1] * width + tensor_kp0[:, 0]).long()

        pred_kp_n1 = raw_pix_coords[bs_i].reshape((-1, 2))[selected_idx]
        dis_x_y = pred_kp_n1 - tensor_kp1
        abs_diff_x_y = torch.abs(dis_x_y)
        l1_x_y = abs_diff_x_y.mean(0) 
        l1 = (l1_x_y[0] / width + l1_x_y[1] / height) / 2
        loss += l1
    loss /= bs
    return loss

def compute_matcher_errors_from_correspondences(correspondences, sample, width, height, bs, device, confidence=0.0, delta=0):
    """Computation of matcher error between reprojected points and pseudo labels
    """
    # convert sample to pixel_coords
    raw_pix_coords = sample / 2 + 0.5
    raw_pix_coords[..., 0] *= width - 1
    raw_pix_coords[..., 1] *= height - 1
    loss = 0
    # calculate matcher error for each image in the batch
    for bs_i in range(bs):
        if confidence > 0:
            mask = correspondences[bs_i]['confidence'] > confidence
            tensor_kp0 = torch.tensor(correspondences[bs_i]['keypoints0'][mask]).to(device)
            tensor_kp1 = torch.tensor(correspondences[bs_i]['keypoints1'][mask]).to(device)
        else:
            tensor_kp0 = torch.tensor(correspondences[bs_i]['keypoints0']).to(device)
            tensor_kp1 = torch.tensor(correspondences[bs_i]['keypoints1']).to(device)

        selected_idx = (tensor_kp0[:, 1] * width + tensor_kp0[:, 0]).long()

        pred_kp_n1 = raw_pix_coords[bs_i].reshape((-1, 2))[selected_idx] # (n, 2)
        dis_x_y = pred_kp_n1 - tensor_kp1
        abs_diff_x_y = torch.abs(dis_x_y)
        if delta > 0:
            eps = 1e-4
            abs_diff_x_y = torch.where(abs_diff_x_y < delta, abs_diff_x_y * eps, abs_diff_x_y)
        l1_x_y = abs_diff_x_y.mean(0)
        l1 = (l1_x_y[0] / width + l1_x_y[1] / height) / 2
        loss += l1
            
            
    loss /= bs
    return loss
```

#### 文件: `modifyppm.py`

```py
import torch
from torch import nn
from .common import LayerNorm2d
from torch.nn import functional as F


class ModifyPPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(ModifyPPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, bias=False, groups=reduction_dim),
                nn.GELU()
            ))
        self.features = nn.ModuleList(self.features)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False, groups=in_dim),
            nn.GELU(),
        )
        self.proconv = nn.Conv2d(in_dim*2,in_dim,kernel_size=1)

    def forward(self, x):
        x_size = x.size()

        out = [self.local_conv(x)]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        o = torch.cat(out, 1)
        o =self.proconv(o)
        return o
if __name__ == '__main__':
    block = ModifyPPM(64,16,[3,6,9,12])
    input_tensor = torch.rand(1, 64, 64, 64)
    output = block(input_tensor)
    print(input_tensor.size(), output.size())
    total_params = sum(p.numel() for p in block.parameters())
    print(f"模型参数总量：{total_params / 1e6:.2f} M")  # 以百万 (M) 为单位

```

#### 文件: `newSAFM.py`

```py
import torch
import torch.nn as nn
import torch.nn.functional as F



class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.feature_conv(x)
        gates = self.sigmoid(self.gate_conv(x))
        return features * gates




class DynamicDepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # 使用全局平均池化获得通道描述
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 生成动态卷积核的全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels * kernel_size * kernel_size, bias=False)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        # 生成每个通道对应的动态卷积核（每个通道独立）
        context = self.global_pool(x).view(b, c)
        dynamic_kernels = self.fc(context)  # shape: [b, c * k*k]
        dynamic_kernels = dynamic_kernels.view(b, c, self.kernel_size, self.kernel_size)
        # 对每个样本、每个通道单独进行卷积操作
        out = []
        for i in range(b):
            # 使用 groups=c 实现 depthwise 卷积
            out.append(F.conv2d(x[i:i+1], dynamic_kernels[i], padding=self.padding, groups=c))
        out = torch.cat(out, dim=0)
        return out





class MultiPoolingFusionSAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        
        # 对于每个尺度，构造两个分支池化后的卷积处理模块
        self.max_conv = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(self.n_levels)]
        )
        self.avg_conv = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(self.n_levels)]
        )

        self.dynamic_convs = nn.ModuleList(
            [DynamicDepthwiseConv2d(chunk_dim, kernel_size=3) for _ in range(self.n_levels)]
        )

        self.gated_convs = nn.ModuleList(
            [GatedConv2d(chunk_dim, chunk_dim, kernel_size=3, padding=1) for _ in range(n_levels)]
        )
        # 用1×1卷积融合拼接后两倍通道的特征
        self.fuse_conv = nn.ModuleList(
            [nn.Conv2d(2 * chunk_dim, chunk_dim, 1, 1, 0) for _ in range(self.n_levels)]
        )
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        # 将特征在通道上切分为 n_levels 个部分
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // (2 ** i), w // (2 ** i))
                # 分别计算最大池化和平均池化
                max_pool = F.adaptive_max_pool2d(xc[i], p_size)
                avg_pool = F.adaptive_avg_pool2d(xc[i], p_size)
                # 分别经过各自的卷积
                max_feat = self.gated_convs[i](max_pool)
                avg_feat = self.gated_convs[i](avg_pool)
                # 拼接融合
                fused = torch.cat([max_feat, avg_feat], dim=1)
                fused = self.fuse_conv[i](fused)
                # 上采样恢复到原始尺寸
                s = F.interpolate(fused, size=(h, w), mode='nearest')
            else:
                # 第一尺度直接卷积处理，不做池化融合
                s = self.gated_convs[i](xc[i])
            out.append(s)
        # 聚合各尺度特征
        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

if __name__ == '__main__':
    input_tensor = torch.randn(3, 36, 64, 64)  # b, c, h, w
    model = MultiPoolingFusionSAFM(dim=36, n_levels=4)
    output = model(input_tensor)
    print(output.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量：{total_params / 1e6:.2f} M")  # 以百万 (M) 为单位

```

#### 文件: `newema.py`

```py
import torch
from torch import nn
import torch.nn.functional as F

class ImprovedEMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(ImprovedEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        # 多种池化方式
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
        # 用膨胀卷积替换普通的3x3卷积，膨胀率可调
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=2, dilation=2)
        # 新增1x1卷积用于融合全局池化结果
        self.fuse_conv = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # [b*groups, c//groups, h, w]

        # 多尺度池化：平均池化与最大池化
        x_avg = self.agp(group_x)
        x_max = self.max_pool(group_x)
        pooled = x_avg + x_max  # [b*groups, c//groups, 1, 1]
        # 融合全局描述
        global_mod = self.fuse_conv(pooled)  # [b*groups, c//groups, 1, 1]
        # 将全局描述扩展后调制输入特征
        group_x = group_x * global_mod.expand_as(group_x)
        
        # 沿高度和宽度方向分别1D池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 引入温度参数控制 Sigmoid 的锐化程度
        x1 = self.gn(group_x * (x_h.sigmoid() * self.temperature) * (x_w.permute(0, 1, 3, 2).sigmoid() * self.temperature))
        x2 = self.conv3x3(group_x)
        
        # 计算注意力权重
        x11 = self.softmax((self.agp(x1).reshape(b * self.groups, -1, 1) / self.temperature).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax((self.agp(x2).reshape(b * self.groups, -1, 1) / self.temperature).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return out

if __name__ == '__main__':
    block = ImprovedEMA(64, factor=8)
    input_tensor = torch.rand(1, 64, 64, 64)
    output = block(input_tensor)
    print(input_tensor.size(), output.size())
    total_params = sum(p.numel() for p in block.parameters())
    print(f"模型参数总量：{total_params / 1e6:.2f} M")  # 以百万 (M) 为单位

```

#### 文件: `pose_cnn.py`

```py
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

```

#### 文件: `pose_decode_litemono.py`

```py
from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_


class PoseDecoderV2(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoderV2, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

```

#### 文件: `pose_decoder.py`

```py
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
```

#### 文件: `pose_intrinsic_decoder.py`

```py
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
```

#### 文件: `resnet_encoder.py`

```py
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
```

#### 文件: `shvit.py`

```py
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import itertools
import numpy as np

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath, to_2tuple

# from mmcv_custom import load_checkpoint, _load_checkpoint, load_state_dict
# from mmdet.utils import get_root_logger
# from mmdet.models.builder import BACKBONES
# from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
_BatchNorm = nn.modules.batchnorm._BatchNorm

class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""

    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))

        return x


class BasicBlock(torch.nn.Module):
    def __init__(self, dim, qk_dim, pdim, type):
        super().__init__()
        if type == "s":  # for later stages
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
            self.mixer = Residual(SHSA(dim, qk_dim, pdim))
            self.ffn = Residual(FFN(dim, int(dim * 2)))
        elif type == "i":  # for early stages
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
            self.mixer = torch.nn.Identity()
            self.ffn = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))


class Partial_ViT_Exp(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 frozen_stages=0,
                 in_chans=3,
                 embed_dim=[128, 256, 384],
                 partial_dim=[32, 64, 96],
                 qk_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 types=["s", "s", "s"],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 pretrained=None,
                 distillation=False, ):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1),
                                               torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1),
                                               torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1))

        resolution = img_size // patch_size
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build SHViT blocks
        for i, (ed, kd, pd, dpth, do, t) in enumerate(
                zip(embed_dim, qk_dim, partial_dim, depth, down_ops, types)):
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(BasicBlock(ed, kd, pd, t))
            if do[0] == 'subsample':
                # Build SHViT downsample block
                # ('Subsample' stride)
                blk = eval('self.blocks' + str(i + 2))
                blk.append(
                    torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                                        Residual(FFN(embed_dim[i], int(embed_dim[i] * 2))), ))
                blk.append(PatchMerging(*embed_dim[i:i + 2]))

                blk.append(torch.nn.Sequential(
                    Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1])),
                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2))), ))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        if pretrained is not None:
            self.init_weights(pretrained=pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(pretrained, map_location='cpu')

            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file {filename}')
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            model_state_dict = self.state_dict()
            # bicubic interpolate attention_biases if not match

            rpe_idx_keys = [
                k for k in state_dict.keys() if "attention_bias_idxs" in k]
            for k in rpe_idx_keys:
                print("deleting key: ", k)
                del state_dict[k]

            relative_position_bias_table_keys = [
                k for k in state_dict.keys() if "attention_biases" in k]
            for k in relative_position_bias_table_keys:
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = model_state_dict[k]
                nH1, L1 = relative_position_bias_table_pretrained.size()
                nH2, L2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    logger.warning(f"Error in loading {k} due to different number of heads")
                else:
                    if L1 != L2:
                        print("resizing key {} from {} * {} to {} * {}".format(k, L1, L1, L2, L2))
                        # bicubic interpolate relative_position_bias_table if not match
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.view(1, nH1, S1, S1), size=(S2, S2),
                            mode='bicubic')
                        state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                            nH2, L2)

            load_state_dict(self, state_dict, strict=False, logger=logger)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(Partial_ViT_Exp, self).train(mode)
        self._freeze_stages()
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        # x = self.patch_embed(x)
        # outs = []
        # x = self.blocks1(x)
        # outs.append(x)
        # x = self.blocks2(x)
        # outs.append(x)
        # x = self.blocks3(x)
        # outs.append(x)
        # return tuple(outs)

        # 1. 逐层运行 patch_embed (分辨率: 1/1 -> 1/2 -> 1/4 -> 1/8 -> 1/16)
        # patch_embed 包含: [0:Conv, 1:ReLU, 2:Conv, 3:ReLU, 4:Conv, 5:ReLU, 6:Conv]
        outs = []
        for i, layer in enumerate(self.patch_embed):
            x = layer(x)
            if i == 1:  # 执行完第1次下采样+ReLU
                feat_1_2 = x  # 1/2 尺度
            if i == 3:  # 执行完第2次下采样+ReLU
                feat_1_4 = x  # 1/4 尺度
            if i == 5:  # 执行完第3次下采样+ReLU
                feat_1_8 = x  # 1/8 尺度

        # 此时 x 是 1/16 尺度 (patch_embed 运行完毕)

        # 2. 运行后续 Block
        x = self.blocks1(x)
        feat_1_16 = x  # 1/16 尺度

        x = self.blocks2(x)
        feat_1_32 = x  # 1/32 尺度 (包含内部 PatchMerging 的下采样)

        # 为了兼容 Monodepth2 的 5 层结构，我们取到 1/32 为止
        # 最后的 blocks3 (1/64) 暂时不用，或者如果你想用，就替换掉 1/32

        # 3. 严格按照 [1/2, 1/4, 1/8, 1/16, 1/32] 的顺序放入列表
        # 这对应 DepthDecoder 中的 input_features[0] 到 [4]
        return (feat_1_2, feat_1_4, feat_1_8, feat_1_16, feat_1_32)

shvit_s4 = {
    'img_size': 256,
    'patch_size': 16,
    'embed_dim': [224, 336, 448],
    'depth': [4, 7, 6],
    'partial_dim': [48, 72, 96],
    'types': ["i", "s", "s"]
}


# @BACKBONES.register_module()
def shvit_s4(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None,
             model_cfg=shvit_s4):
    model = Partial_ViT_Exp(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    return model


class SHViTEncoder(nn.Module):
    def __init__(self, model_type="shvit_s1", height=256, width=320):
        super(SHViTEncoder, self).__init__()

        # 确保 S1 的参数完全匹配下载的权重
        configs = {
            'shvit_s1': {
                # 'img_size': (height, width),
                'img_size': height,
                'patch_size': 16,
                'embed_dim': [128, 224, 320],  # S1 必须是这个通道数
                'depth': [2, 4, 5],
                'partial_dim': [32, 48, 68],
                'types': ["i", "s", "s"]
            }
        }

        cfg = configs[model_type]
        from .shvit import Partial_ViT_Exp
        self.encoder = Partial_ViT_Exp(**cfg)

        # --- 核心：计算 5 级通道数，与 forward 的输出一一对应 ---
        e0 = cfg['embed_dim'][0]
        # 按照 patch_embed 内部的 embed_dim // 8, // 4, // 2 推导
        self.num_ch_enc = np.array([
            e0 // 8,  # 1/2 尺度 -> 16
            e0 // 4,  # 1/4 尺度 -> 32
            e0 // 2,  # 1/8 尺度 -> 64
            e0,  # 1/16 尺度 -> 128
            cfg['embed_dim'][1]  # 1/32 尺度 -> 256
        ])
        # 最终 S1 的 num_ch_enc 为 [16, 32, 64, 128, 256]
        # ----------------------------------------------------

    def load_pretrained(self, weight_path):
        """
        显式加载权重函数
        使用方法: encoder.load_pretrained('/your/path/shvit_s1.pth')
        """
        if not os.path.isfile(weight_path):
            print(f"=> [Error] 找不到权重文件: {weight_path}")
            return

        print(f"=> 正在显式加载 SHViT 权重: {weight_path}")
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

        # 1. 自动定位字典
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

        # 2. 清理键名（适配从 mmcv 等框架导出的权重）
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[7:]
            if name.startswith('backbone.'): name = name[9:]

            # 过滤掉不属于编码器的层（例如分类头）
            if any(x in name for x in ['head', 'fc', 'classifier']):
                continue
            new_state_dict[name] = v

        # 3. 加载
        msg = self.encoder.load_state_dict(new_state_dict, strict=False)
        print(f"=> 加载完成！结果报告: {msg}")

    def forward(self, x):
        x = (x - 0.45) / 0.225
        return list(self.encoder(x))
```

#### 文件: `swiftformer.py`

```py
# The code is adapted from SwiftFormer.
# SwiftFormer GitHub: https://github.com/Amshaker/SwiftFormer
# SwiftFormer paper: https://arxiv.org/abs/2303.15446

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
import einops


def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2d(in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        query_weight = query @ self.w_g
        A = query_weight * self.scale_factor

        A = A.softmax(dim=-1)

        G = torch.sum(A * query, dim=1)

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )

        out = self.Proj(G * key) + query

        out = self.final(out)

        return out


class SwiftFormerLocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class SwiftFormerEncoder(nn.Module):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2) EfficientAdditiveAttention, and (3) MLP block.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.local_representation = SwiftFormerLocalRepresentation(dim=dim, kernel_size=3, drop_path=0.,
                                                                   use_layer_scale=True)
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        x = self.local_representation(x)
        B, C, H, W = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(
                    0, 3, 1, 2))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C)).reshape(B, H, W, C).permute(0, 3, 1, 2))
            x = x + self.drop_path(self.linear(x))
        return x


def Stage(dim, index, layers, mlp_ratio=4.,
          act_layer=nn.GELU,
          drop_rate=.0, drop_path_rate=0.,
          use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    """
    Implementation of each SwiftFormer stages. Here, SwiftFormerEncoder used as the last block in all stages, while ConvEncoder used in the rest of the blocks.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

        if layers[index] - block_idx <= vit_num:
            blocks.append(SwiftFormerEncoder(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value))

        else:
            blocks.append(ConvEncoder(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3))

    blocks = nn.Sequential(*blocks)
    return blocks


class SwiftFormer(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 act_layer=nn.GELU,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 vit_num=1,
                 **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.patch_embed = stem(3, self.embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = Stage(self.embed_dims[i], i, layers, mlp_ratio=mlp_ratios,
                          act_layer=act_layer,
                          drop_rate=drop_rate,
                          drop_path_rate=drop_path_rate,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init_value=layer_scale_init_value,
                          vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or self.embed_dims[i] != self.embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=self.embed_dims[i], embed_dim=self.embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            layer = nn.BatchNorm2d(self.embed_dims[i_emb])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x, x0):
        outs = [x0]
        for idx, block in enumerate(self.network):
            x = block(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs

    def forward(self, x):
        x0 = self.patch_embed[2](self.patch_embed[1]((self.patch_embed[0](x))))
        x = self.patch_embed[5](self.patch_embed[4]((self.patch_embed[3](x0))))
        x = self.forward_tokens(x, x0)
        return x


SwiftFormer_width = {
    'XS': [48, 56, 112, 220],
    'S': [48, 64, 168, 224],
}

SwiftFormer_depth = {
    'XS': [3, 3, 6, 4],
    'S': [3, 3, 9, 6],
}


class SwiftFormer_XS(SwiftFormer):
    def __init__(self, **kwargs):
        super().__init__(
            layers=SwiftFormer_depth['XS'],
            embed_dims=SwiftFormer_width['XS'],
            downsamples=[True, True, True, True],
            vit_num=1,
            rate=0.2,
            drop_path_rate=0.2,
            **kwargs)


class SwiftFormer_S(SwiftFormer):
    def __init__(self, **kwargs):
        super().__init__(
            layers=SwiftFormer_depth['S'],
            embed_dims=SwiftFormer_width['S'],
            downsamples=[True, True, True, True],
            vit_num=1,
            drop_rate=0.4,
            drop_path_rate=0.4,
            **kwargs)


if __name__ == '__main__':
    model = SwiftFormer_S()

    input2 = torch.randn(8, 3, 256, 320)

    y12 = []
    y12 = model(input2)
    print(len(y12))
    print(y12[0].shape)
    print(y12[1].shape)
    print(y12[2].shape)
    print(y12[3].shape)
    print(y12[4].shape)
```

