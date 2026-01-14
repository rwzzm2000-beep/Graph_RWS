# Graph_RWS/helper/utils.py

import os
import yaml
import json
import torch
from datetime import datetime


# -------------------- 配置加载函数 --------------------
def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# -------------------- 设备设置函数 --------------------
def setup_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BCSR GNN training!")
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    return device, gpu_name


# -------------------- 路径管理封装函数 --------------------
def setup_run_env(config, ds_name):
    """
    配置本次运行的保存环境：
    根据 config 中的 save_model, save_path, save_result, result_path 决定
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "Sample" if config.get('sampling', False) else "Full"
    run_id = f"{ds_name}_{mode_str}_{timestamp}"

    # 1. 准备模型保存目录
    current_model_dir = None
    if config.get('save_model', False):
        base_save_path = config.get('save_path', 'Graph_RWS/save_models/')
        current_model_dir = os.path.join(base_save_path, run_id)
        os.makedirs(current_model_dir, exist_ok=True)
        print(f"  [Save] Model directory created: {current_model_dir}")

    # 2. 准备结果保存文件路径
    result_file_path = None
    if config.get('save_result', True):
        base_result_path = config.get('result_path', 'Graph_RWS/results/')
        os.makedirs(base_result_path, exist_ok=True)
        result_file_path = os.path.join(base_result_path, f"{run_id}.json")
        print(f"  [Save] Result file path set to: {result_file_path}")

    return current_model_dir, result_file_path


# -------------------- 模型保存函数 --------------------
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)