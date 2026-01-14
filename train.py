# Graph_RWS/train.py
import os
import time
import argparse
import warnings
import json

# 忽略 outdated 库的警告
warnings.filterwarnings("ignore", category=UserWarning, module="outdated")

# 导入我们封装好的模块
from .models import GCN_BCSR
from .data import load_dataset, BCSRGraph
from .sampler import BCSRSampler
from .helper import *
from .helper.engine import (
    train_epoch_full,
    train_epoch_node_sampling,
    train_epoch_dps_static,
    train_epoch_baps,
    evaluate, 
    PartitionDataLoader,
)

import torch                    
import torch.nn.functional as F


def main():
    # torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser(description='BCSR GNN Training Framework')
    parser.add_argument('--config', type=str, default='configs/cora.yaml', help='Path to configuration file')        
    args = parser.parse_args()

    # 初始化全局 Stage 计时器
    global_timer = StageTimer()

    # [Stage 1] Environment & Configuration
    global_timer.start_stage("[Stage 1] Environment & Configuration")
    device, gpu_name = setup_device()
    print(f"Running on GPU: {gpu_name}\n")

    config = load_config(args.config)

    enable_save_model = config.get('save_model', False)
    enable_save_result = config.get('save_result', True)
    sampling_batch_size = config.get('node', {}).get('batch_size', config.get('batch_size', 1024))

    # [New] 获取训练模式
    train_mode = config.get('train_mode', 'node')
    print(f"[Config] Train Mode: {train_mode}")

    # 判断是否需要采样器 (DPS, BAPS, node 都需要 sampler 对象)
    enable_sampling = (train_mode != 'full')

    log_config_info(config)

    # [Stage 2] Data Loading
    # 通过 data_loader 统一加载：支持 DGL / OGB → BCSR
    global_timer.start_stage("[Stage 2] Data Loading & Preprocessing")
    bundle = load_dataset(config, device)
    partition_book = bundle.partition_book

    ds_name = log_dataset_info(bundle, config)

    # 初始化保存环境（路径生成与创建）
    current_model_dir, result_file_path = setup_run_env(config, ds_name)

    print("\n[Memory Strategy] Configuring Memory Layout...")
    
    # 1. 读取配置
    use_uva = config.get('use_uva', False)
    #如果是 DPS 模式且没有开启 UVA，通常是因为数据太大（如 Reddit 42GB），默认让特征留在 CPU
    # 这样可以把宝贵的显存留给 Graph Structure 和动态生成的子图
    force_cpu_features = (train_mode == 'DPS') and (not use_uva)
    
    # 2. 处理特征 (Features) 和 标签 (Labels)
    if use_uva or force_cpu_features:
        print(f"  -> Feature Mode: CPU Resident (Pinned). UVA={use_uva}")
        
        # [Fix] 增加检查：如果特征已经在 GPU 上，则跳过 Pin 操作
        feats = bundle.features
        if feats.is_cuda:
            print("  -> [Warning] Features are ALREADY on GPU. Skipping pin_memory().")
            # 已经在 GPU 上了，不需要做任何操作，直接使用
        else:
            # 在 CPU 上，执行 Pinning
            feats = feats.share_memory_().pin_memory()
            
        bundle = bundle._replace(
            features=feats,
            labels=bundle.labels.to(device)
        )
    else:
        print("  -> Feature Mode: GPU Resident (High Performance).")
        try:
            # 尝试全量放 GPU
            bundle = bundle._replace(
                features=bundle.features.to(device),
                labels=bundle.labels.to(device)
            )
        except RuntimeError as e:
            print(f"  -> [Error] GPU OOM: {e}")
            print("  -> Suggestion: Set 'use_uva: True' in config to keep features on CPU.")
            raise e

    # 3. 处理图结构 (Graph Structure)
    # [关键] Index-Based DPS 必须要求图结构在 GPU 上，以便进行高速切图
    if train_mode == 'DPS':
        if not bundle.bcsr_full.window_offset.is_cuda:
            print("  -> [DPS Requirement] Moving Graph Structure to GPU...")
            bundle.bcsr_full.to(device)
    elif train_mode == 'node' or train_mode == 'full':
        # 其他模式如果显存允许，也可以把图放 GPU
        if not use_uva: # UVA 模式下通常图也在 GPU，这里做个兜底
             if not bundle.bcsr_full.window_offset.is_cuda:
                try:
                    print("  -> Moving Graph Structure to GPU...")
                    # 使用 _replace 更新 bundle，并接收 .to() 的返回值
                    bundle = bundle._replace(
                        bcsr_full=bundle.bcsr_full.to(device)
                    )
                except Exception as e:
                    print(f"  -> [Warning] Failed to move graph to GPU: {e}")
                    pass

    # 4. 处理索引 (Indices)
    # 训练/验证/测试集的索引都很小，直接放 GPU 方便计算 Mask
    bundle = bundle._replace(
        train_idx=bundle.train_idx.to(device),
        val_idx=bundle.val_idx.to(device),
        test_idx=bundle.test_idx.to(device)
    )

    # 5. 更新局部变量引用
    features = bundle.features
    
    # 6. 清理不再需要的 SuperMap (DPS 预处理产物)
    if hasattr(bundle, 'super_map') and bundle.super_map is not None:
        # Index-Based 模式下，训练阶段不需要 SuperMap，为了省内存可以清理或移到 CPU
        bundle = bundle._replace(super_map=None)

    # [Debug] 检查特征矩阵
    print(f"\n[Debug] Checking features for NaN/Inf...")
    if torch.isnan(bundle.features).any() or torch.isinf(bundle.features).any():
        print("!!!!!! FATAL ERROR: Features contain NaN or Inf immediately after loading! !!!!!!")
        # 打印出问题的前几个值
        mask = torch.isnan(bundle.features) | torch.isinf(bundle.features)
        print(f"Indices of Bad Values: {torch.nonzero(mask, as_tuple=False)[:10]}")
        exit(1)
    else:
        print("[Debug] Features are clean (No NaN/Inf).")

    # [Stage 3] Model Initialization
    # 构造模型
    global_timer.start_stage("[Stage 3] Model Initialization")
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

    # 准备图数据结构
    if enable_sampling:
        p_book_arg = partition_book if train_mode in ['DPS', 'BAPS'] else None

        # 采样模式：初始化采样器，Model 输入将在 train_epoch 中动态生成
        sampler = BCSRSampler(
            fanouts=config.get('fanouts'),
            tile_rows=int(config.get('tile_rows', 16)),
            tile_cols=int(config.get('tile_cols', 8)),
            warps_per_block=int(config.get('warps_per_block', 8)),
            partition_book=p_book_arg,
            verbose=True
        )
        graphs_input = [bundle.bcsr_full] # 仅作为源数据传入
    else:
        # 全图模式：直接复制 full graph
        sampler = None
        graphs_input = [bundle.bcsr_full] * model.num_layers

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get('lr', 1e-3)),
        weight_decay=float(config.get('weight_decay', 0.0))
    )

    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

    # 初始化 Loaders 和 Mask
    dps_loader = None
    baps_loader = None
    train_bool_mask = None

    # 只有 DPS 和 BAPS 需要 train_bool_mask 来过滤子图
    if train_mode in ['DPS', 'BAPS']:
        # [Modified] Index-Based Storage 适配逻辑
        # 现在的 features 是原始大小 (N_orig)，而不是膨胀后的 SuperGraph 大小
        
        num_orig_nodes = bundle.features.shape[0]
        train_bool_mask = torch.zeros(num_orig_nodes, dtype=torch.bool, device=device)
        
        if train_mode == 'DPS' and hasattr(bundle, 'super_map') and bundle.super_map is not None:
            # 关键修正：DPS 模式下的 bundle.train_idx 是 SuperGraph 的索引
            # 我们需要利用 super_map 将其映射回 Global ID 才能生成正确的 Mask
            # super_map 在 CPU，train_idx 在 GPU，需同步设备
            super_map_dev = bundle.super_map.to(device)
            global_train_idx = super_map_dev[bundle.train_idx]
            train_bool_mask[global_train_idx] = True
            del super_map_dev # 省显存
        else:
            # BAPS 或 旧版 DPS (bundle.train_idx 已经是 Global ID)
            train_bool_mask[bundle.train_idx] = True

    if train_mode == 'DPS':
        print(f"[DPS_STATIC] Initializing Partition Loader...")
        if partition_book is None:
            raise ValueError("DPS_STATIC mode requires partition_book (METIS). Check data_loader logic.")
        
        # 尝试读取配置，没有则使用默认值
        dps_static_cfg = config.get('BAPS', {})
        part_batch_size = dps_static_cfg.get('batch_partition_size', 10)
        
        # 直接从 partition_book 获取分区总数，确保一致性
        num_parts = len(partition_book)
        
        # 复用 baps_loader 变量，因为它本质上就是一个 PartitionDataLoader
        dps_loader = PartitionDataLoader(num_partitions=num_parts, batch_size=part_batch_size, device=device)
        
    elif train_mode == 'BAPS':
        print(f"[BAPS] Initializing Partition Loader...")
        baps_cfg = config.get('BAPS', {})
        # 注意: BAPS 模式需要 partition_book 存在
        if partition_book is None:
                raise ValueError("BAPS mode requires METIS preprocessing. Check data_loader logic.")
                
        num_parts = baps_cfg.get('num_partitions', 100)
        part_batch_size = baps_cfg.get('batch_partition_size', 10)
        baps_loader = PartitionDataLoader(num_partitions=num_parts, batch_size=part_batch_size, device=device)

    
    # [Stage 4] Training Loop
    global_timer.start_stage("[Stage 4] Training Process")
    
    # 初始化日志器和计时器
    logger = TableLogger()
    epoch_timer = EpochTimer()

    logger.print_header() # 打印表头

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = None

    patience = int(config.get('patience', 50))

    epoch_durations = []       # 用于记录每个 epoch 的耗时
    train_start_time = time.time() # 记录训练开始的绝对时间

    for epoch in range(int(config.get('epochs', 50))):
        t0 = time.time()

        # ================= [Modified] 训练模式选择 =================
        # 优先级: DPS > BAPS > Point Sampling
        
        if train_mode == 'DPS':
            # --- [Branch 1] DPS Static Mode ---
             train_loss, train_acc = train_epoch_dps_static(
                model=model,
                bcsr_full=bundle.bcsr_full,
                features=bundle.features,
                labels=bundle.labels,
                train_bool_mask=train_bool_mask,
                optimizer=optimizer,
                device=device,
                partition_loader=dps_loader, 
                sampler=sampler,
                partition_book=partition_book
            )
            
        elif train_mode == 'BAPS':
            # --- [Branch 2] BAPS Mode ---
            train_loss, train_acc = train_epoch_baps(
                model=model,
                bcsr_full=bundle.bcsr_full,
                features=bundle.features,
                labels=bundle.labels,
                train_bool_mask=train_bool_mask,
                optimizer=optimizer,
                device=device,
                baps_loader=baps_loader,
                sampler=sampler,
                partition_book=partition_book
            )
            
        elif train_mode == 'node':
            # --- [Branch 3] Node Sampling ---
            train_loss, train_acc = train_epoch_node_sampling(
                model=model,
                graphs_for_model=graphs_input,
                features=bundle.features,
                labels=bundle.labels,
                train_idx=bundle.train_idx,
                optimizer=optimizer,
                device=device,
                sampler=sampler,
                batch_size=sampling_batch_size,
                tile_rows=int(config.get('tile_rows', 16)),
                timer=epoch_timer,
                pipeline=config.get('pipeline', True)
            )

        elif train_mode == 'full':
            # --- [Branch 4] Full Graph ---
            train_loss, train_acc = train_epoch_full(
                model=model,
                graphs_for_model=graphs_input,
                features=bundle.features,
                labels=bundle.labels,
                train_idx=bundle.train_idx,
                optimizer=optimizer,
                device=device,
                timer=epoch_timer
            )

        dt = time.time() - t0
        epoch_durations.append(dt)

        # Evaluation
        val_acc = None
        if epoch % int(config.get('eval_interval', 5)) == 0:
            val_acc = evaluate(model, bundle.bcsr_full, bundle.features, bundle.labels, bundle.val_idx, device)
            
            # Update Scheduler
            if scheduler:
                scheduler.step(val_acc)
            
            # Checkpoint Logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                if enable_save_model and current_model_dir:
                    ckpt_name = f"model_epoch_{epoch}.pt"
                    save_full_path = os.path.join(current_model_dir, ckpt_name)
                    save_checkpoint(model, optimizer, epoch, train_loss, save_full_path)
                    best_model_path = save_full_path

        # 打印表格行
        logger.log_epoch(epoch, dt, train_loss, train_acc, val_acc, epoch_timer.get_epoch_stats())

        torch.cuda.empty_cache()
        
        # Early Stopping
        if epoch - best_epoch > patience:
            print("\nEarly stopping triggered!")
            break

    logger.print_avg_epoch_time()

    # [Stage 5] Final Testing
    global_timer.start_stage("[Stage 5] Final Evaluation")

    # 从记录的路径加载最佳模型
    if enable_save_model and best_model_path and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from: {best_model_path}")
        print(f"  (Epoch {checkpoint['epoch']}, Val Acc: {best_val_acc*100:.2f}%)")
    elif enable_save_model:
        print("Warning: Best model checkpoint not found (maybe first epoch was best?).")

    test_acc = evaluate(model, bundle.bcsr_full, bundle.features, bundle.labels, bundle.test_idx, device)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f} %")

    # 调用采样器的平均时间统计
    if enable_sampling and sampler is not None:
        sampler.print_avg_timing()
    
    total_training_time = time.time() - train_start_time
    actual_epochs = len(epoch_durations)
    avg_epoch_time = sum(epoch_durations) / actual_epochs if actual_epochs > 0 else 0.0

    if enable_save_result and result_file_path:
        results = {
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'test_acc': test_acc,
            'config': config,
            'model_dir': current_model_dir,
            'best_model_path': best_model_path,
            'total_training_time_sec': total_training_time,
            'avg_epoch_time_sec': avg_epoch_time,
            'total_epochs_run': actual_epochs
        }
        with open(result_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {result_file_path}")
    
    global_timer.end_stage() # 结束最后一个 Stage
    print("\nGraph_RWS Finished.")
    os._exit(0)


if __name__ == '__main__':
    main()
