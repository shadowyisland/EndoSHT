import yaml
import torch
import argparse
import os
from depthnet.model import EstimateDepth


def count_parameters(model, module_name):
    """
    计算单个模型模块的参数数量
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    return {
        'module': module_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6
    }


def print_results(results):
    """
    格式化打印参数统计结果
    """
    print("=" * 80)
    print("Network Parameter Count Results")
    print("=" * 80)
    
    headers = ["Module", "Total Params", "Trainable Params", "Total (M)", "Trainable (M)"]
    print(f"{headers[0]:<25} {headers[1]:<15} {headers[2]:<18} {headers[3]:<10} {headers[4]:<12}")
    print("-" * 80)
    
    total_all = 0
    trainable_all = 0
    
    for result in results:
        print(f"{result['module']:<25} {result['total_params']:<15,} {result['trainable_params']:<18,} "
              f"{result['total_params_millions']:<10.4f} {result['trainable_params_millions']:<12.4f}")
        total_all += result['total_params']
        trainable_all += result['trainable_params']
    
    print("-" * 80)
    print(f"{'Total':<25} {total_all:<15,} {trainable_all:<18,} "
          f"{total_all/1e6:<10.4f} {trainable_all/1e6:<12.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Count network model parameters")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for counting (cpu/cuda)')
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found!")
        return
    
    # 读取配置文件
    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {args.config}")
    print(f"Model Type: {cfgs.get('model_str', 'Unknown')}")
    print(f"Model Name: {cfgs.get('model_name', 'Unknown')}")
    print()
    
    # 创建模型
    model = EstimateDepth(cfgs)
    
    # 将模型移动到指定设备
    model.to_device(args.device)
    
    # 收集所有网络模块
    network_names = model.network_names
    print(f"Found {len(network_names)} network modules: {network_names}")
    print()
    
    # 计算每个模块的参数
    results = []
    for net_name in network_names:
        net_module = getattr(model, net_name)
        result = count_parameters(net_module, net_name)
        results.append(result)
    
    # 添加额外参数统计（如果需要）
    print_results(results)


if __name__ == '__main__':
    main()
