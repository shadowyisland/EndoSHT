import os
import cv2
import glob
import numpy as np
from tqdm import tqdm


# ============================================================
# 配置参数 — 按你的数据集路径修改这里即可
# ============================================================
SRC_ROOT = r"/media/mems509/9b308a11-7150-4494-8f42-71df9385ff43/home/mems509/wjy/MonoLoT-main/data/c3vd_v2/imgs"
DST_ROOT = r"./gt_depth_visualized"          # 输出根目录
GAMMA = 0.6                                   # 伽马校正系数 (0~1: 提亮暗部)
COLORMAP = cv2.COLORMAP_TURBO                 # 颜色映射
MAX_DEPTH_BITS = 16                           # GT tiff 位深度
EXT_DEPTH = "_depth.tiff"                     # GT 深度文件后缀
# ============================================================


def render_depth(disp, gamma=0.6, colormap=cv2.COLORMAP_TURBO):
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    disp = 1.0 - np.power(disp, gamma)
    disp = (disp * 255).astype(np.uint8)
    return cv2.applyColorMap(disp, colormap)


def main():
    print(f"SRC 根目录:  {SRC_ROOT}")
    print(f"DST 根目录:  {DST_ROOT}")
    print(f"深度文件后缀: {EXT_DEPTH}\n")

    total = 0

    for dirpath, _, filenames in os.walk(SRC_ROOT):
        depth_files = sorted([f for f in filenames if f.endswith(EXT_DEPTH)])
        if not depth_files:
            continue

        rel_dir = os.path.relpath(dirpath, SRC_ROOT)
        dst_dir = os.path.join(DST_ROOT, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        prefix = rel_dir if rel_dir != "." else "根目录"
        print(f"[{prefix}]  {len(depth_files)} 张深度图")

        for fname in tqdm(depth_files, desc=f"  {prefix}", leave=False):
            src_path = os.path.join(dirpath, fname)
            gt = cv2.imread(src_path, -1)
            if gt is None:
                print(f"    警告: 无法读取 {src_path}")
                continue

            gt = gt.astype(np.float32) / (2 ** MAX_DEPTH_BITS)
            vis = render_depth(gt, gamma=GAMMA, colormap=COLORMAP)

            out_name = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(dst_dir, out_name), vis)
            total += 1

    print(f"\n完成 — 共处理 {total} 张 GT 深度图，保存至: {os.path.abspath(DST_ROOT)}")


if __name__ == "__main__":
    main()