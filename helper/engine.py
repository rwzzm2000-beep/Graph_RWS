# Graph_RWS/helper/engine.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GCN_BCSR
from ..sampler import BCSRSampler
from .timer import EpochTimer
from ..data.data_loader import (
    WindowWiseDataLoader,
    MegaBatchDataLoader,
    PartitionDataLoader
)



def compute_loss_and_metrics(logits, labels):
    """
    Returns:
        loss: Tensor
        correct_count: Tensor (GPU上的标量) 或 int
        num_samples: int
    """
    if labels.dim() > 1: # Multi-label
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        pred = (logits > 0).float()
        correct_count = (pred == labels).sum() 
        num_samples = labels.numel() 
    else: # Single-label
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=1)
        correct_count = (pred == labels).sum() 
        num_samples = labels.size(0)
        
    return loss, correct_count, num_samples


# -------------------- 训练/评估（纯 BCSR，全图） --------------------
def train_epoch_full(model: 'GCN_BCSR',
                     graphs_for_model,
                     features: torch.Tensor,
                     labels: torch.Tensor,
                     train_idx: torch.Tensor,
                     optimizer: torch.optim.Optimizer,
                     device,
                     timer: EpochTimer = None):
    """
    [Mode 1] 全图训练 (Full Graph Training)
    直接对整张图进行前向传播和反向传播。
    """
    model.train()

    # 重置计时器
    if timer: timer.reset()

    # 全图模式下确保特征在 GPU
    if features.device != device:
         features = features.to(device) 
    
    optimizer.zero_grad()
    
    # 1. Forward
    logits = model(graphs_for_model, features)

    # 2. Slice (只计算训练集节点的 Loss)
    batch_labels = labels[train_idx]
    batch_logits = logits[train_idx]
    
    # 3. Loss & Metric
    loss, correct_count, num_samples = compute_loss_and_metrics(batch_logits, batch_labels)
    acc = correct_count / num_samples if num_samples > 0 else 0.0

    # 4. Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), acc


def train_epoch_node_sampling(model: 'GCN_BCSR',
                               graphs_for_model,
                               features: torch.Tensor,
                               labels: torch.Tensor,
                               train_idx: torch.Tensor,
                               optimizer: torch.optim.Optimizer,
                               device,
                               sampler: BCSRSampler,
                               batch_size: int = 1024,
                               tile_rows: int = 16,
                               timer: EpochTimer = None,
                               pipeline: bool = True):
    """
    [Mode 2] 节点采样训练 (Node Sampling / Mini-batch Training)
    支持普通的串行采样和基于 CUDA Stream 的流水线并行采样。
    """
    model.train()
    if timer: timer.reset()

    # 准备变量累加器
    total_loss_tensor = torch.zeros(1, device=device)
    total_correct_tensor = torch.zeros(1, device=device)
    total_samples = 0
    num_batches = 0

    # === Pipeline Mode: 开启流水线并行 ===
    if pipeline:
        # 1. 准备 Loader 和 Stream
        train_idx_cpu = train_idx.cpu() if train_idx.is_cuda else train_idx
        loader = WindowWiseDataLoader(train_idx_cpu, batch_size=batch_size, tile_rows=tile_rows, drop_last=False)
        bcsr_full = graphs_for_model[0] if isinstance(graphs_for_model, list) else graphs_for_model
        
        sample_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        data_queue = queue.Queue(maxsize=1)
        is_uva_mode = (features.device.type == 'cpu')

        # 2. 生产者线程 (负责采样和特征提取)
        def producer():
            with torch.cuda.stream(sample_stream):
                for batch_seeds in loader:
                    batch_seeds = batch_seeds.to(device, non_blocking=True)
                    batch_seeds, _ = torch.sort(batch_seeds)

                    # 采样
                    subgraphs = sampler.sample_layers(bcsr_full, batch_seeds)
                    batch_features = None
                    
                    # 特征提取 (如果是 UVA 模式，利用 transfer_stream 掩盖耗时)
                    if is_uva_mode:
                        transfer_stream.wait_stream(sample_stream)
                        with torch.cuda.stream(transfer_stream):
                            g0 = subgraphs[0]
                            input_nodes_gpu = g0.original_col_indices
                            input_nodes_cpu = input_nodes_gpu.cpu()
                            batch_features_cpu = features[input_nodes_cpu]
                            batch_features = batch_features_cpu.to(device, non_blocking=True)
                            
                            # 更新子图索引，标记为 Local Input
                            new_local_indices = torch.arange(input_nodes_gpu.size(0), device=device, dtype=torch.int32)
                            g0.original_col_indices = new_local_indices
                            g0.is_local_input = True
                    
                    data_queue.put((batch_seeds, subgraphs, batch_features))
            data_queue.put(None) # 结束信号

        worker = threading.Thread(target=producer, daemon=True)
        worker.start()
            
        # 3. 消费者循环 (负责训练)
        while True:
            timer.start_batch()
            timer.record('Data') 
            item = data_queue.get()
            if item is None: break
            
            batch_seeds, subgraphs, batch_features = item
            
            # 同步流
            torch.cuda.current_stream().wait_stream(transfer_stream)
            torch.cuda.current_stream().wait_stream(sample_stream)
            timer.record('Sample')

            optimizer.zero_grad()
            curr_features = batch_features if batch_features is not None else features
            logits = model(subgraphs, curr_features)
            timer.record('Fwd')
            
            # --- Row Mapping & Loss ---
            output_graph = subgraphs[-1]
            actual_active_windows = output_graph.active_windows
            seed_windows = torch.div(batch_seeds, tile_rows, rounding_mode='floor')
            sorted_active, perm = torch.sort(actual_active_windows)
            sorted_indices = torch.searchsorted(sorted_active, seed_windows)
            window_indices = perm[sorted_indices]
            
            local_rows = window_indices * tile_rows + (batch_seeds % tile_rows)
            local_rows = torch.clamp(local_rows, max=logits.shape[0] - 1)

            batch_logits = logits[local_rows]
            batch_labels = labels[batch_seeds]
            
            loss, correct_count, num_samples = compute_loss_and_metrics(batch_logits, batch_labels)
            
            loss.backward()
            timer.record('Bwd')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            timer.record('Step')
            
            with torch.no_grad():
                total_loss_tensor += loss.detach() 
                total_correct_tensor += correct_count
                total_samples += num_samples
                num_batches += 1
            timer.end_batch()

        worker.join()

    # === Serial Mode: 普通串行模式 (Pipeline=False) ===
    else:
        loader = DataLoader(train_idx, batch_size=batch_size, shuffle=True, drop_last=False)
        bcsr_full = graphs_for_model[0] if isinstance(graphs_for_model, list) else graphs_for_model
        
        for batch_seeds in loader:
            timer.start_batch()
            batch_seeds = batch_seeds.to(device)
            batch_seeds, _ = torch.sort(batch_seeds) 
            timer.record('Data')
            
            subgraphs = sampler.sample_layers(bcsr_full, batch_seeds)
            timer.record('Sample')
            
            optimizer.zero_grad()
            logits = model(subgraphs, features)
            timer.record('Fwd')
            
            # --- Row Mapping & Loss ---
            output_graph = subgraphs[-1]
            actual_active_windows = output_graph.active_windows
            seed_windows = torch.div(batch_seeds, tile_rows, rounding_mode='floor')
            sorted_active, perm = torch.sort(actual_active_windows)
            sorted_indices = torch.searchsorted(sorted_active, seed_windows)
            window_indices = perm[sorted_indices]
            
            local_rows = window_indices * tile_rows + (batch_seeds % tile_rows)
            local_rows = torch.clamp(local_rows, max=logits.shape[0] - 1)
            
            batch_logits = logits[local_rows]
            batch_labels = labels[batch_seeds]
            
            loss, correct_count, num_samples = compute_loss_and_metrics(batch_logits, batch_labels)
            
            loss.backward()
            timer.record('Bwd')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            timer.record('Step')
            
            with torch.no_grad():
                total_loss_tensor += loss.detach() * (num_samples / batch_labels.numel() if batch_labels.dim() > 1 else 1)
                total_correct_tensor += correct_count
                total_samples += num_samples
                num_batches += 1
            timer.end_batch()

    # 计算平均值
    avg_loss = total_loss_tensor.item() / num_batches
    avg_acc = total_correct_tensor.item() / total_samples

    return avg_loss, avg_acc


def train_epoch_baps(model, 
                     bcsr_full, 
                     features: torch.Tensor, 
                     labels: torch.Tensor, 
                     train_bool_mask: torch.Tensor, 
                     optimizer: torch.optim.Optimizer, 
                     device, 
                     baps_loader, 
                     sampler, 
                     partition_book):
    """
    执行一个 BAPS 训练 Epoch:
    Partition Sampling -> Zero-Copy Slicing -> Subgraph Training
    """
    model.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = 0
    
    for batch_pids in baps_loader:
        # 1. 采样 & 切片
        subgraph = sampler.sample_partitions(bcsr_full, batch_pids)
        
        batch_indices_list = []
        pids_list = batch_pids.tolist()
        
        for pid in pids_list:
            info = partition_book[pid]
            batch_indices_list.append(
                torch.arange(info['start_idx'], info['end_idx'], device=device)
            )
        
        batch_global_indices = torch.cat(batch_indices_list)
        batch_inputs = features[batch_global_indices]
        batch_labels = labels[batch_global_indices]
        batch_train_mask = train_bool_mask[batch_global_indices]
        
        if not batch_train_mask.any():
            continue

        # 2. 前向传播
        optimizer.zero_grad()
        graphs_list = [subgraph] * model.num_layers
        logits = model(graphs_list, batch_inputs)
        
        # 3. 过滤出训练节点
        loss_logits = logits[batch_train_mask]
        loss_labels = batch_labels[batch_train_mask]
        
        # [修改] 使用统一辅助函数 (修复了之前统计部分的 Bug)
        loss, correct_count, num_samples = compute_loss_and_metrics(loss_logits, loss_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 4. 统计
        with torch.no_grad():
            # 注意: 这里 loss 是 batch 的 mean，为了算 epoch avg，这里加权累加
            # 单标签时 num_samples = nodes，多标签时 num_samples = elements
            # 这里的 loss.item() 是 per-element 的 mean
            total_loss += loss.item() * num_samples 
            total_correct += correct_count
            total_samples += num_samples
            num_batches += 1
    
    # 计算 Epoch 平均指标
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_acc


def train_epoch_dps_static(model, 
                           bcsr_full, 
                           features: torch.Tensor, 
                           labels: torch.Tensor, 
                           train_bool_mask: torch.Tensor, 
                           optimizer: torch.optim.Optimizer, 
                           device, 
                           partition_loader, 
                           sampler, 
                           partition_book):
    """
    [DPS Index-Based Mode]
    基于索引的动态切图训练 (High Performance)。
    
    流程：
    1. CPU: 从 PartitionLoader 获取 Batch Partition IDs。
    2. CPU: 从 PartitionBook 查表得到 Window IDs，合并并传输到 GPU。
    3. GPU (Stream): 调用 sampler.repack_subgraph 实时重组物理连续的子图。
    4. GPU (Stream): 利用重组后的 original_node_ids 切取特征 (支持 UVA)。
    5. GPU (Main): 模型训练。
    """
    model.train()
    
    # 统计变量 (使用 Tensor 避免频繁 CPU-GPU 同步)
    total_loss_tensor = torch.zeros(1, device=device)
    total_correct_tensor = torch.zeros(1, device=device)
    total_samples = 0
    num_batches = 0
    
    # ---------------- [Pipeline Setup] ----------------
    import queue
    import threading
    
    # 创建专用流用于 采样(Repack) 和 数据传输
    sample_stream = torch.cuda.Stream()
    data_queue = queue.Queue(maxsize=2) # 缓冲 2 个 Batch 足以掩盖延迟
    stop_event = threading.Event()
    
    # 生产者线程：准备数据 (Index -> Subgraph -> Features)
    def producer():
        # 让生产者在独立的 CUDA Stream 中运行
        with torch.cuda.stream(sample_stream):
            for batch_pids in partition_loader:
                if stop_event.is_set(): break
                
                try:
                    # 1. [CPU] 准备 Window IDs
                    # partition_book 现在是 List[Tensor]，根据 ID 取出对应的 Window 列表
                    pids_list = batch_pids.tolist()
                    batch_windows_list = [partition_book[pid] for pid in pids_list]
                    
                    # 合并 Window IDs 并非阻塞传输到 GPU
                    # 这里的 batch_window_ids 是重组子图唯一的“配方”
                    batch_window_ids = torch.cat(batch_windows_list).to(device, non_blocking=True)
                    
                    # 2. [GPU] 实时切结构 (Repack Subgraph)
                    # bcsr_full 必须是 GPU 上的原图
                    # 这步操作极快 (纯 GPU 访存 + 偏移量计算)
                    subgraph = sampler.repack_subgraph(bcsr_full, batch_window_ids)
                    
                    # 3. [GPU/UVA] 切特征 (Feature Slicing)
                    # repack_subgraph 会自动生成 original_node_ids (GPU Tensor)
                    # 如果 features 是 Pinned CPU Memory (UVA)，这行代码会触发 PCIe 传输
                    # 如果 features 在 GPU，则是片内拷贝
                    global_nids = subgraph.original_node_ids
                    
                    # PyTorch 不支持 cpu_tensor[gpu_indices]，我们需要把索引转回 CPU
                    # 由于 features 是 Pinned Memory，.to(device, non_blocking=True) 会非常快 (DMA 传输)
                    batch_inputs = features[global_nids.cpu()].to(device, non_blocking=True)
                    
                    # Labels 同理 (如果 labels 在 GPU 则不需要 cpu()，但为了通用性...)
                    # 你的 Data Bundle 中 labels 已经在 GPU 了，所以直接切
                    if labels.is_cuda:
                        batch_labels = labels[global_nids]
                    else:
                        batch_labels = labels[global_nids.cpu()].to(device, non_blocking=True)

                    # Mask 同理
                    if train_bool_mask.is_cuda:
                         batch_mask = train_bool_mask[global_nids]
                    else:
                         batch_mask = train_bool_mask[global_nids.cpu()].to(device, non_blocking=True)
                    
                    # 4. 入队
                    data_queue.put((subgraph, batch_inputs, batch_labels, batch_mask))
                    
                except Exception as e:
                    print(f"Producer Error in DPS Loop: {e}")
                    stop_event.set()
                    break
                    
        data_queue.put(None) # 结束信号

    # 启动后台线程
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    
    # ---------------- [Consumer Loop: Training] ----------------
    try:
        while True:
            # 获取数据
            item = data_queue.get()
            if item is None: break
            
            subgraph, batch_inputs, batch_labels, batch_mask = item
            
            torch.cuda.current_stream().wait_stream(sample_stream)
            
            if not batch_mask.any():
                del subgraph, batch_inputs, batch_labels, batch_mask
                continue
            
            # ========= [Debug 探针 1: 检查输入特征] =========
            if torch.isnan(batch_inputs).any():
                print(f"[Epoch Debug] NaN detected in batch_inputs!")
                print(f"  Input Shape: {batch_inputs.shape}")
                print(f"  Bad Values Count: {torch.isnan(batch_inputs).sum().item()}")
                # 还可以检查一下 batch_window_ids 或 original_node_ids
                raise ValueError("NaN in Input Features")

            # Forward
            optimizer.zero_grad()
            
            if hasattr(subgraph, 'inner_graph'):
                # Layer 0 使用 Halo Graph (N_out x N_in)
                # Layer 1+ 使用 Inner Graph (N_out x N_out)
                graphs_list = [subgraph] + [subgraph.inner_graph] * (model.num_layers - 1)
            else:
                # 兼容旧模式 (BAPS 或无 Halo)
                graphs_list = [subgraph] * model.num_layers
            
            # 开启异常检测（可选，会变慢，但能定位具体的 Layer）
            # with torch.autograd.detect_anomaly():
            logits = model(graphs_list, batch_inputs)
            
            # ========= [Debug 探针 2: 检查模型输出] =========
            if torch.isnan(logits).any():
                print(f"[Epoch Debug] NaN detected in logits (Forward Pass Failed)!")
                print(f"  Logits Shape: {logits.shape}")
                print(f"  Batch Inputs NaN?: {torch.isnan(batch_inputs).any()}")
                raise ValueError("NaN in Model Output")
            
            # ========= [Fix] 维度对齐修复 =========
            # Logits 维度: [N_internal, Classes]
            # Batch Mask/Labels 维度: [N_internal + N_halo, ...]
            # 必须先切片，只取 Internal 部分进行 Loss 计算
            
            n_internal = logits.size(0)
            
            # 1. 确保 Mask 和 Labels 与 Logits 长度一致
            # 注意：repack_subgraph 保证了 Internal Nodes 总是排在最前面
            target_mask = batch_mask[:n_internal]
            target_labels = batch_labels[:n_internal]
            
            # 2. 检查是否有训练节点
            if not target_mask.any():
                del subgraph, batch_inputs, batch_labels, batch_mask, logits
                continue

            # 3. 应用 Mask
            loss_logits = logits[target_mask]
            loss_labels = target_labels[target_mask]
            
            loss, correct_count, batch_num_samples = compute_loss_and_metrics(loss_logits, loss_labels)
            
            if torch.isnan(loss):
                 print("[Epoch Debug] Loss is NaN!")
                 print(f"  Logits slice NaN?: {torch.isnan(loss_logits).any()}")
                 raise ValueError("NaN Loss")

            loss.backward()
            
            # ========= [Debug 探针 3: 检查梯度] =========
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"[Epoch Debug] NaN gradient detected in {name}!")
                    raise ValueError(f"NaN Gradient in {name}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            
            # Stats Accumulation
            with torch.no_grad():
                total_loss_tensor += loss.detach() * batch_num_samples
                total_correct_tensor += correct_count
                total_samples += batch_num_samples
                num_batches += 1
            
            # Clean up immediately to save GPU memory
            del subgraph, batch_inputs, batch_labels, batch_mask, logits, loss
            
    except KeyboardInterrupt:
        stop_event.set()
        raise
    finally:
        thread.join()

    # 计算平均指标
    if total_samples > 0:
        avg_loss = total_loss_tensor.item() / total_samples
        avg_acc = total_correct_tensor.item() / total_samples
    else:
        avg_loss, avg_acc = 0.0, 0.0
    
    return avg_loss, avg_acc    


@torch.no_grad()
def evaluate_dps_sampling(model: 'GCN_BCSR',
                          bcsr_full,
                          features: torch.Tensor,
                          labels: torch.Tensor,
                          eval_bool_mask: torch.Tensor,
                          device,
                          partition_loader,
                          sampler,
                          partition_book):
    """
    [DPS Evaluation] 复用 DPS 切图管线做评估，避免全图推理 OOM。
    """
    model.eval()

    total_correct_tensor = torch.zeros(1, device=device)
    total_samples = 0

    import queue
    import threading

    sample_stream = torch.cuda.Stream()
    data_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    def producer():
        with torch.cuda.stream(sample_stream):
            for batch_pids in partition_loader:
                if stop_event.is_set():
                    break
                try:
                    pids_list = batch_pids.tolist()
                    batch_windows_list = [partition_book[pid] for pid in pids_list]
                    batch_window_ids = torch.cat(batch_windows_list).to(device, non_blocking=True)

                    subgraph = sampler.repack_subgraph(bcsr_full, batch_window_ids)

                    global_nids = subgraph.original_node_ids
                    batch_inputs = features[global_nids.cpu()].to(device, non_blocking=True)

                    if labels.is_cuda:
                        batch_labels = labels[global_nids]
                    else:
                        batch_labels = labels[global_nids.cpu()].to(device, non_blocking=True)

                    if eval_bool_mask.is_cuda:
                        batch_mask = eval_bool_mask[global_nids]
                    else:
                        batch_mask = eval_bool_mask[global_nids.cpu()].to(device, non_blocking=True)

                    data_queue.put((subgraph, batch_inputs, batch_labels, batch_mask))
                except Exception as e:
                    print(f"Producer Error in DPS Eval: {e}")
                    stop_event.set()
                    break
        data_queue.put(None)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    try:
        while True:
            item = data_queue.get()
            if item is None:
                break
            subgraph, batch_inputs, batch_labels, batch_mask = item
            torch.cuda.current_stream().wait_stream(sample_stream)

            if not batch_mask.any():
                del subgraph, batch_inputs, batch_labels, batch_mask
                continue

            if hasattr(subgraph, 'inner_graph'):
                graphs_list = [subgraph] + [subgraph.inner_graph] * (model.num_layers - 1)
            else:
                graphs_list = [subgraph] * model.num_layers

            logits = model(graphs_list, batch_inputs)
            n_internal = logits.size(0)
            target_mask = batch_mask[:n_internal]
            target_labels = batch_labels[:n_internal]

            if not target_mask.any():
                del subgraph, batch_inputs, batch_labels, batch_mask, logits
                continue

            eval_logits = logits[target_mask]
            eval_labels = target_labels[target_mask]
            _, correct_count, batch_num_samples = compute_loss_and_metrics(eval_logits, eval_labels)

            total_correct_tensor += correct_count
            total_samples += batch_num_samples

            del subgraph, batch_inputs, batch_labels, batch_mask, logits
    finally:
        thread.join()

    if total_samples > 0:
        acc = (total_correct_tensor / total_samples).item()
    else:
        acc = 0.0
    return acc


@torch.no_grad()
def evaluate(model: 'GCN_BCSR',
             bcsr_full,
             features: torch.Tensor,
             labels: torch.Tensor,
             eval_idx: torch.Tensor,
             device):
    model.eval()
    logits = model.inference(bcsr_full, features)
    
    logits_eval = logits[eval_idx]
    labels_eval = labels[eval_idx]
    
    # 使用统一辅助函数
    _, correct_count, num_samples = compute_loss_and_metrics(logits_eval, labels_eval)

    if num_samples > 0:
        acc = (correct_count / num_samples).item()
    else:
        acc = 0.0
        
    return acc


