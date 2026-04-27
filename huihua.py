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