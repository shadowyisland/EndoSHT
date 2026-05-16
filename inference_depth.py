import os
import csv
import torch
import cv2
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from depthnet.utils import load_yaml
from depthnet.model import EstimateDepth
from depthnet.networks.layers import disp_to_depth

# --- 实验配置参数 ---
IMAGE_PATH = r"/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/data/c3vd_v2/imgs"
CONFIG_PATH = r"/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/experiments/c3vd_v2/monodepth2/shvit_edfm_v2.yml"
WEIGHTS_DIR = r"/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/results_shvit_edfm_v2/shvit_edfm_v2/models/weights_best"
SAVE_DIR = r"./inference_results/test1"

# --- [优化1] 排序指标可配置 ---
SORT_METRIC = "abs_rel"      # abs_rel | rmse | rmse_log | a1 | composite (综合分)
TOP_K_PER_DIR = 10            # 每个子目录保留前 K 张
TOP_K_GLOBAL = 20             # 全局跨目录保留前 K 张
SAVE_ALL_DEPTH = False        # True=保存所有深度图 | False=只保存最优结果，大幅节省磁盘
GEN_COMPARISON = True         # 生成 原图|GT|预测 同框对比图
BATCH_SIZE = 8                # batch 推理加速（>=2）
GAMMA_CORRECT = 0.6           # 深度渲染伽马校正系数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_errors(gt, pred):
    """计算深度估计的各项指标。"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt + 1e-7) - np.log(pred + 1e-7)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return {
        "abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse,
        "rmse_log": rmse_log, "a1": a1, "a2": a2, "a3": a3
    }


def render_depth(disp, gamma=0.6):
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    disp = np.power(disp, gamma)
    disp = (disp * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
    return disp_color


def composite_score(m):
    """[优化2] 综合分数: 加权组合多个指标, 兼顾准确率和误差"""
    return m["abs_rel"] + 0.5 * m["sq_rel"] + 0.1 * m["rmse"] + 3.0 * (1 - m["a1"])


def make_comparison_image(original_path, gt_path, pred_vis, metrics_text=""):
    """[优化3] 生成原图 + Ground Truth + 预测深度 同框对比图"""
    original = cv2.imread(original_path)
    if original is None:
        original = np.zeros_like(pred_vis)
    original = cv2.resize(original, (pred_vis.shape[1], pred_vis.shape[0]))

    gt_depth = cv2.imread(gt_path, -1) if os.path.exists(gt_path) else None
    if gt_depth is not None:
        gt_depth = gt_depth.astype(np.float32) / (2 ** 16)
        gt_vis = render_depth(gt_depth, gamma=GAMMA_CORRECT)
        gt_vis = cv2.resize(gt_vis, (pred_vis.shape[1], pred_vis.shape[0]))
    else:
        gt_vis = np.zeros_like(pred_vis)

    h, w = pred_vis.shape[:2]
    bar = np.zeros((h, 4, 3), dtype=np.uint8) + 80
    comparison = np.hstack([original, bar, gt_vis, bar, pred_vis])

    if metrics_text:
        label_y = h - 10
        for i, line in enumerate(metrics_text.split("\n")):
            cv2.putText(comparison, line, (5, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    labels = ["Input", "GT", "Predicted"]
    for i, label in enumerate(labels):
        x = i * (w + 4)
        cv2.putText(comparison, label, (x + 5, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return comparison


class ImageFolderDataset(Dataset):
    """[优化4] 批量推理用 Dataset, 支持 batch inference"""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path, img.size


def main():
    device = torch.device(DEVICE)

    # 1. 加载配置信息
    print(f"-> 正在加载配置文件: {CONFIG_PATH}")
    cfgs = load_yaml(CONFIG_PATH)
    cfgs.update({'device': device})

    # 推理时不需要位姿网络和内参网络
    cfgs['not_load_nets'] = ["net_pose_encoder", "net_pose_decoder", "net_depth_intrinsics"]

    # 2. 初始化模型
    model = EstimateDepth(cfgs)

    # 3. 加载权重文件
    print(f"-> 正在从目录加载权重: {WEIGHTS_DIR}")
    weights_path = Path(WEIGHTS_DIR)
    state_dict = {}

    # 确定需要加载的网络部分
    available_nets = [k for k in model.network_names if k not in cfgs['not_load_nets']]

    for net_name in available_nets:
        pth_file = weights_path / f"{net_name}.pth"
        if pth_file.exists():
            state_dict[net_name] = torch.load(pth_file, map_location=device)
        else:
            print(f"警告: 未找到 {pth_file}，跳过加载。")

    model.load_model_state(state_dict)
    model.to_device(device)
    model.set_eval()

    # 4. 获取所有包含图片的子目录
    print(f"-> 正在扫描目录结构: {IMAGE_PATH}")
    target_dirs = []
    for root, dirs, files in os.walk(IMAGE_PATH):
        # 检查当前目录下是否有图片文件
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in files):
            target_dirs.append(root)

    if not target_dirs:
        print(f"!!! 在 {IMAGE_PATH} 下未找到任何包含图片的文件夹。")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    height, width = cfgs.get('height', 256), cfgs.get('width', 320)
    transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])

    # [优化5] 全局排名收集器
    all_global_results = []

    # 5. 按子目录逐个进行处理
    for current_dir in target_dirs:
        rel_dir = os.path.relpath(current_dir, IMAGE_PATH)
        display_dir = "根目录" if rel_dir == "." else rel_dir
        print(f"\n[目录开始] -> 正在处理: {display_dir}")

        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            image_files.extend(glob.glob(os.path.join(current_dir, ext)))
        if not image_files:
            continue

        target_save_dir = os.path.join(SAVE_DIR, rel_dir)
        os.makedirs(target_save_dir, exist_ok=True)

        # --- [优化6] Batch 推理 ---
        dataset = ImageFolderDataset(image_files, transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        current_dir_results = []

        with torch.no_grad():
            for batch_tensors, batch_paths, batch_sizes in dataloader:
                batch_tensors = batch_tensors.to(device)
                features = model.net_depth_encoder(batch_tensors)
                outputs = model.net_depth_decoder(features)
                disp = outputs[("disp", 0)]
                pred_disps, _ = disp_to_depth(disp, cfgs["min_depth"], cfgs["max_depth"])
                pred_disps_np = pred_disps.cpu().numpy()[:, 0]

                for i, img_path in enumerate(batch_paths):
                    img_basename = os.path.basename(img_path)
                    original_width, original_height = batch_sizes[0][i].item(), batch_sizes[1][i].item()
                    pred_disp_np = pred_disps_np[i]

                    vis_depth = render_depth(pred_disp_np, gamma=GAMMA_CORRECT)
                    vis_depth = cv2.resize(vis_depth, (original_width, original_height))

                    # 评估指标
                    gt_path = img_path.replace("_color.png", "_depth.tiff").replace(".jpg", "_depth.tiff")
                    metrics = None
                    if os.path.exists(gt_path):
                        gt_depth = cv2.imread(gt_path, -1)
                        if gt_depth is not None:
                            gt_depth = gt_depth.astype(np.float32) / (2 ** 16)
                            pred_disp_resized = cv2.resize(pred_disp_np, (original_width, original_height))
                            pred_depth = 1.0 / (pred_disp_resized + 1e-7)
                            mask = gt_depth > 0
                            pred_depth_m, gt_depth_m = pred_depth[mask], gt_depth[mask]
                            ratio = np.median(gt_depth_m) / np.median(pred_depth_m)
                            pred_depth_m *= ratio
                            metrics = compute_errors(gt_depth_m, pred_depth_m)

                    entry = {
                        "abs_rel": metrics["abs_rel"] if metrics else 1e9,
                        "rmse": metrics["rmse"] if metrics else 1e9,
                        "a1": metrics["a1"] if metrics else -1,
                        "sq_rel": metrics["sq_rel"] if metrics else 1e9,
                        "rmse_log": metrics["rmse_log"] if metrics else 1e9,
                        "a2": metrics["a2"] if metrics else -1,
                        "a3": metrics["a3"] if metrics else -1,
                        "img_basename": os.path.splitext(img_basename)[0],
                        "dir": display_dir,
                        "img_path": img_path,
                        "gt_path": gt_path,
                        "vis_img": vis_depth,
                    }
                    if metrics:
                        entry["composite"] = composite_score(metrics)
                    current_dir_results.append(entry)

                    # [优化7] 可选: 只保存Top-K, 跳过全量保存
                    if SAVE_ALL_DEPTH:
                        save_path = os.path.join(target_save_dir,
                                                 f"{os.path.splitext(img_basename)[0]}_depth.png")
                        cv2.imwrite(save_path, vis_depth)

        # --- 处理完当前目录，排序及保存 ---
        if current_dir_results:
            key_fn = _sort_key_fn()
            reverse = SORT_METRIC in ("a1", "a2", "a3")
            current_dir_results.sort(key=key_fn, reverse=reverse)

            sub_best_dir = os.path.join(target_save_dir, "best")
            os.makedirs(sub_best_dir, exist_ok=True)

            for i in range(min(TOP_K_PER_DIR, len(current_dir_results))):
                res = current_dir_results[i]
                metric_val = key_fn(res)
                if GEN_COMPARISON:
                    metrics_str = f"{SORT_METRIC}={metric_val:.4f}  a1={res['a1']:.3f}  rmse={res['rmse']:.4f}"
                    comp = make_comparison_image(res["img_path"], res["gt_path"], res["vis_img"], metrics_str)
                    best_name = f"rank{i+1:02d}_{SORT_METRIC}_{metric_val:.4f}_{res['img_basename']}.png"
                    cv2.imwrite(os.path.join(sub_best_dir, best_name), comp)
                else:
                    best_name = f"rank{i+1:02d}_{SORT_METRIC}_{metric_val:.4f}_{res['img_basename']}.png"
                    cv2.imwrite(os.path.join(sub_best_dir, best_name), res["vis_img"])

            all_global_results.extend(current_dir_results)

            _export_csv(current_dir_results, target_save_dir, display_dir)

        print(f"[目录完成] -> {display_dir} 处理完毕 (共 {len(current_dir_results)} 张)")

    # --- [优化8] 全局跨目录排名 ---
    if all_global_results:
        _save_global_best(all_global_results)

    print(f"\n-> 所有任务处理完成。结果镜像保存至: {os.path.abspath(SAVE_DIR)}")


def _sort_key_fn():
    """根据 SORT_METRIC 返回排序 key 函数"""
    if SORT_METRIC == "composite":
        return lambda r: r.get("composite", 1e9)
    return lambda r: r.get(SORT_METRIC, 1e9)


def _export_csv(results, save_dir, dir_label):
    """[优化9] 导出当前目录的指标 CSV"""
    csv_path = os.path.join(save_dir, f"metrics_{dir_label.replace('/', '_')}.csv")
    fields = ["rank", "dir", "img_basename", "abs_rel", "sq_rel", "rmse", "rmse_log",
              "a1", "a2", "a3", "composite"]
    key_fn = _sort_key_fn()
    reverse = SORT_METRIC in ("a1", "a2", "a3")
    sorted_results = sorted(results, key=key_fn, reverse=reverse)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(sorted_results):
            row = {k: r.get(k, "") for k in fields}
            row["rank"] = i + 1
            writer.writerow(row)
    print(f"   -> CSV 指标已导出: {csv_path}")


def _save_global_best(all_results):
    """[优化10] 全局跨目录最优排名 & 汇总 CSV"""
    global_best_dir = os.path.join(SAVE_DIR, "_global_best")
    os.makedirs(global_best_dir, exist_ok=True)

    key_fn = _sort_key_fn()
    reverse = SORT_METRIC in ("a1", "a2", "a3")
    sorted_all = sorted(all_results, key=key_fn, reverse=reverse)

    for i in range(min(TOP_K_GLOBAL, len(sorted_all))):
        res = sorted_all[i]
        src_dir = os.path.join(SAVE_DIR, res["dir"] if res["dir"] != "根目录" else ".", "best")
        for fname in os.listdir(src_dir) if os.path.isdir(src_dir) else []:
            if res["img_basename"] in fname:
                import shutil
                shutil.copy2(os.path.join(src_dir, fname),
                             os.path.join(global_best_dir,
                                          f"global_rank{i+1:02d}_{fname}"))
                break

    _export_csv(sorted_all, global_best_dir, "global")
    print(f"\n-> 全局 Top-{TOP_K_GLOBAL} 已保存至: {global_best_dir}")


if __name__ == "__main__":
    main()