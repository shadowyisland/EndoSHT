import os
import torch
import cv2
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from depthnet.utils import load_yaml
from depthnet.model import EstimateDepth
from depthnet.networks.layers import disp_to_depth

# --- 实验配置参数 ---
# 在这里指定你的数据集位置和模型权重路径
IMAGE_PATH = r"/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/data/c3vd_v2/imgs"  # 图片根目录
CONFIG_PATH = r"/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/experiments/c3vd_v2/monodepth2/shvit_edfm_v2.yml"
WEIGHTS_DIR = r"/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/results_shvit_edfm_v2/shvit_edfm_v2/models/weights_best"
SAVE_DIR = r"./inference_results/test1"  # 所有推理结果保存路径（将在此目录下镜像原始结构）
TOP_K = 20  # 每个子文件夹挑选效果最好的前 20 张图片
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


def render_depth(disp):
    """将视差图归一化并上色，用于可视化。"""
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    disp = np.power(disp, 0.6)  # 伽马校正，使深处细节更清晰
    disp = (disp * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_TURBO)
    return disp_color


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

    # 5. 按子目录逐个进行处理
    for current_dir in target_dirs:
        rel_dir = os.path.relpath(current_dir, IMAGE_PATH)
        # 如果是根目录本身，rel_dir 会是 "."
        display_dir = "根目录" if rel_dir == "." else rel_dir
        print(f"\n[目录开始] -> 正在处理: {display_dir}")

        # 获取当前目录下的所有图片（非递归）
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            image_files.extend(glob.glob(os.path.join(current_dir, ext)))
        
        if not image_files:
            continue

        # 准备当前目录的保存路径
        target_save_dir = os.path.join(SAVE_DIR, rel_dir)
        os.makedirs(target_save_dir, exist_ok=True)
        
        current_dir_results = []

        # 推理当前目录下的所有图片
        with torch.no_grad():
            for img_path in image_files:
                img_basename = os.path.basename(img_path)
                
                # 加载并预处理
                input_image = Image.open(img_path).convert('RGB')
                original_width, original_height = input_image.size
                input_tensor = transform(input_image).unsqueeze(0).to(device)

                # 模型预测
                features = model.net_depth_encoder(input_tensor)
                outputs = model.net_depth_decoder(features)

                # 获取深度
                disp = outputs[("disp", 0)]
                pred_disp, _ = disp_to_depth(disp, cfgs["min_depth"], cfgs["max_depth"])
                pred_disp_np = pred_disp.cpu().numpy()[0, 0]

                # 渲染并保存普通结果
                vis_depth = render_depth(pred_disp_np)
                vis_depth = cv2.resize(vis_depth, (original_width, original_height))
                
                save_path = os.path.join(target_save_dir, f"{os.path.splitext(img_basename)[0]}_depth.png")
                cv2.imwrite(save_path, vis_depth)

                # --- 评估指标 ---
                gt_path = img_path.replace("_color.png", "_depth.tiff").replace(".jpg", "_depth.tiff")
                if os.path.exists(gt_path):
                    gt_depth = cv2.imread(gt_path, -1) / (2 ** 16)
                    if gt_depth is not None:
                        pred_disp_resized = cv2.resize(pred_disp_np, (original_width, original_height))
                        pred_depth = 1.0 / (pred_disp_resized + 1e-7)
                        mask = gt_depth > 0
                        pred_depth_m, gt_depth_m = pred_depth[mask], gt_depth[mask]
                        ratio = np.median(gt_depth_m) / np.median(pred_depth_m)
                        pred_depth_m *= ratio
                        metrics = compute_errors(gt_depth_m, pred_depth_m)
                        
                        current_dir_results.append({
                            "abs_rel": metrics["abs_rel"],
                            "img_basename": os.path.splitext(img_basename)[0],
                            "vis_img": vis_depth
                        })

        # 处理完当前目录后，立即挑选并保存该目录的最优结果
        if current_dir_results:
            print(f"   -> 正在挑选该目录的最优 Top-{TOP_K}...")
            current_dir_results.sort(key=lambda x: x["abs_rel"])
            
            sub_best_dir = os.path.join(target_save_dir, "best")
            os.makedirs(sub_best_dir, exist_ok=True)
            
            for i in range(min(TOP_K, len(current_dir_results))):
                res = current_dir_results[i]
                best_name = f"rank{i+1:02d}_absrel_{res['abs_rel']:.4f}_{res['img_basename']}.png"
                cv2.imwrite(os.path.join(sub_best_dir, best_name), res["vis_img"])
            
        print(f"[目录完成] -> {display_dir} 处理完毕。")

    print(f"\n-> 所有任务处理完成。结果镜像保存至: {os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()