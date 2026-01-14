# Graph_RWS/test.py
import os
import argparse
import torch
import warnings

# 忽略 outdated 库的警告
warnings.filterwarnings("ignore", category=UserWarning, module="outdated")

# 导入项目模块
# 注意：作为模块运行时 (python -m Graph_RWS.test)，使用相对导入
from .models import GCN_BCSR
from .data import load_dataset
from .helper import setup_device, load_config, log_config_info
from .helper.engine import evaluate

def main():
    parser = argparse.ArgumentParser(description='BCSR GNN Testing/Inference Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file used for training')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the saved model checkpoint (.pt file)')
    args = parser.parse_args()

    # 1. 环境与配置
    device, gpu_name = setup_device()
    print(f"Running Inference on GPU: {gpu_name}\n")

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    log_config_info(config)

    # 2. 加载数据
    # 必须加载与训练时相同的数据集，以保证特征维度和图结构一致
    print("Loading dataset...")
    bundle = load_dataset(config, device)
    
    print(f"Dataset Loaded: {bundle.features.shape[0]} nodes, {bundle.num_classes} classes.")

    # 3. 初始化模型
    # 必须使用与训练时完全相同的超参数构造模型结构
    print("Initializing model architecture...")
    model = GCN_BCSR(
        in_feats=bundle.features.shape[1],
        hidden_size=int(config.get('hidden_size', 128)),
        num_classes=int(config.get('num_classes', bundle.num_classes)),
        num_layers=len(config.get('fanouts', [10, 10])),
        dropout=float(config.get('dropout', 0.5)),
        residual=bool(config.get('residual', False)),
        warps_per_block=int(config.get('warps_per_block', 8)),
        use_checkpoint=config.get('use_checkpoint', True)
    ).to(device)

    # 4. 加载权重
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    print(f"Loading model weights from: {checkpoint_path}")
    # 使用 weights_only=True 更安全，如果报错改成 False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # 兼容处理：检查 checkpoint 是否包含 key 'model_state_dict' (常见的保存格式)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  -> Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # 假设整个 checkpoint 就是 state_dict
        model.load_state_dict(checkpoint)
    
    print("Model loaded successfully.")

    # 5. 执行测试
    print("Starting evaluation on TEST set...")
    model.eval()
    
    # 使用全图推理进行评估
    test_acc = evaluate(
        model, 
        bundle.bcsr_full, 
        bundle.features, 
        bundle.labels, 
        bundle.test_idx, 
        device
    )

    print("=" * 60)
    print(f"Final Test Accuracy: {test_acc * 100:.2f} %")
    print("=" * 60)

if __name__ == '__main__':
    main()