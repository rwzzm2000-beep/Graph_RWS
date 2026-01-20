# Graph_RWS/data/data_loader.py
"""
统一数据加载入口（DGL / OGB → BCSRGraph）
- 负责数据下载/读取与基本拆包（features/labels/splits）
- 图格式转换统一委托给 BCSRGraph 提供的 classmethod（from_dgl）
- train.py 只需调用 load_dataset(config, device) 获得 DataBundle
"""

from typing import Dict, Any, NamedTuple
import importlib
import torch
from torch.nn import functional as F
import numpy as np
import os
import scipy.sparse.csgraph
from typing import Tuple

from .bcsr import BCSRGraph
from ..sampler import BCSRSampler



class DataBundle(NamedTuple):
    bcsr_full: BCSRGraph
    features: torch.Tensor
    labels: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    num_classes: int
    partition_book: Any = None
    super_map: Any = None  # 存储 SuperGraph ID -> Original Global ID 的映射
    static_scores: torch.Tensor = None
    window_part_ids: torch.Tensor = None
    part_ranges: torch.Tensor = None
    hacs_enabled: bool = False


class WindowWiseDataLoader:
    """
    基于 Window 的数据加载器。
    
    逻辑：
    1. 预处理：将所有 train_idx 按照其所属的 Window ID 进行分组。
    2. Shuffle：每个 Epoch 开始时，对 Window ID 进行随机打乱 (而不是对节点打乱)。
    3. Batching：依次取出 Window，直到凑够 batch_size 个节点，返回这批节点。
    
    优势：
    - 保证一个 Batch 内，如果选中了某个 Window，该 Window 内的所有训练节点都会被包含。
    - 极大提升 Window 利用率 (从 1/16 提升到 16/16 或 Window内实际训练节点数)。
    """
    def __init__(self, train_idx, batch_size, tile_rows=16, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = train_idx.device
        
        # 1. 计算所有训练节点的 Window ID
        # 为了加速分组，我们将计算移到 CPU (只在初始化时做一次)
        train_idx_cpu = train_idx.cpu()
        wids = train_idx_cpu.div(tile_rows, rounding_mode='floor')
        
        # 2. 排序并分组
        # 通过 argsort 让属于同一个 Window 的节点聚在一起
        sort_idx = torch.argsort(wids)
        self.sorted_train_idx = train_idx_cpu[sort_idx]
        sorted_wids = wids[sort_idx]
        
        # 3. 获取每个 Window 的边界 (Start, Count)
        # unique_consecutive: 找出排好序的 Window ID 中每个 ID 出现的次数
        unique_wids, counts = torch.unique_consecutive(sorted_wids, return_counts=True)
        
        # 构建 (start_index, node_count) 列表
        # 例如: Window 0 从索引 0 开始有 3 个点，Window 5 从索引 3 开始有 16 个点...
        self.window_info = [] 
        curr = 0
        for count in counts.tolist():
            self.window_info.append((curr, count))
            curr += count
            
        self.num_windows = len(self.window_info)
        # print(f"[WindowWiseLoader] Total Training Nodes: {len(train_idx)}, Active Windows: {self.num_windows}")
        
    def __iter__(self):
        # 1. Shuffle Windows (打乱 Window 的顺序)
        perm = torch.randperm(self.num_windows)
        
        batch_nodes_list = []
        current_node_count = 0
        
        # 2. 遍历打乱后的 Window
        for i in range(self.num_windows):
            win_idx = perm[i].item()
            start, count = self.window_info[win_idx]
            
            # 取出该 Window 下的所有训练节点
            nodes = self.sorted_train_idx[start : start + count]
            batch_nodes_list.append(nodes)
            current_node_count += count
            
            # 3. 凑够 Batch Size 后 Yield
            if current_node_count >= self.batch_size:
                # 拼接并转回 GPU
                yield torch.cat(batch_nodes_list).to(self.device)
                
                # 重置
                batch_nodes_list = []
                current_node_count = 0
        
        # 4. 处理剩余数据
        if not self.drop_last and len(batch_nodes_list) > 0:
            yield torch.cat(batch_nodes_list).to(self.device)

    def __len__(self):
        # 估算 Batch 数量
        total_nodes = len(self.sorted_train_idx)
        return (total_nodes + self.batch_size - 1) // self.batch_size


class MegaBatchDataLoader:
    """
    超级分块 DataLoader：
    1. 先将所有训练节点随机打乱 (Global Shuffle)。
    2. 将打乱后的节点切分成多个 Mega-Batches (Pools)。
    3. 在每个 Mega-Batch 内部进行排序 (Local Sort)。
    4. 将排序后的 Mega-Batch 切分成最终的 Mini-Batches。
    
    结果：既保留了 Epoch 级别的随机性，又保证了 Mini-Batch 内部极高的数据局部性。
    """
    def __init__(self, indices, batch_size, mega_batch_factor=50, drop_last=False):
        self.indices = indices
        self.batch_size = batch_size
        # Mega-Batch 大小是 Mini-Batch 的 N 倍 (建议 50-100)
        self.mega_batch_size = batch_size * mega_batch_factor
        self.drop_last = drop_last
        
    def __iter__(self):
        # 1. Global Shuffle: 保证每个 Epoch 训练样本的顺序不同
        perm = torch.randperm(self.indices.size(0), device=self.indices.device)
        shuffled_indices = self.indices[perm]
        
        num_indices = shuffled_indices.size(0)
        
        # 2. Process each Mega-Batch
        for i in range(0, num_indices, self.mega_batch_size):
            # 获取一个 Mega-Batch (Pool)
            pool_end = min(i + self.mega_batch_size, num_indices)
            if self.drop_last and (pool_end - i) < self.batch_size:
                break
                
            pool = shuffled_indices[i:pool_end]
            
            # 3. Sort inside the Pool (关键步骤!)
            # 这会让 ID 相近的节点聚在一起，大幅提升 Window 利用率
            pool, _ = torch.sort(pool)
            
            # 4. Slice into Mini-Batches
            for j in range(0, pool.size(0), self.batch_size):
                batch_end = min(j + self.batch_size, pool.size(0))
                if self.drop_last and (batch_end - j) < self.batch_size:
                    continue
                    
                yield pool[j:batch_end]

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size


class PartitionDataLoader:
    """
    BAPS/DPS 专用加载器 (Block-Aligned Partition Sampling)
    
    [优化策略]: Chunk Shuffle (块状打乱)
    - 保持 Partition ID 的局部连续性，将它们按顺序切分成 Chunks (Batches)。
    - 既然 METIS 生成的 ID 具有空间局部性，相邻 ID 的 Partition 往往物理相邻。
    - 将它们放在同一个 Batch 中，可以最大化内部边比例，最小化 Halo Nodes 数量，
      从而避免 OOM 并保留更多真实的图结构信息。
    - 为了保证训练的随机性，我们在 Epoch 级别随机打乱这些 Chunks 的顺序。
    """
    def __init__(self, num_partitions: int, batch_size: int = 10, drop_last: bool = False, device='cpu'):
        self.num_partitions = num_partitions
        self.batch_size = batch_size # 这里指 "Partitions per Batch"
        self.drop_last = drop_last
        self.device = device
        
        # 保证 partition_ids 是连续的 [0, 1, 2, ..., N-1]
        self.partition_ids = torch.arange(num_partitions, device=device)
        
    def __iter__(self):
        # 1. 准备数据: 按顺序切分为 Chunks (Batches)
        # 例如 batch_size=10: Chunk0=[0..9], Chunk1=[10..19]...
        # 这些 Chunk 内部的 Partition 互为邻居的概率最大。
        pids = self.partition_ids
        chunks = list(torch.split(pids, self.batch_size))
        
        # 2. Batch 级别的 Shuffle
        # 随机打乱 Chunks 的处理顺序，保证 Epoch 间的随机性
        num_chunks = len(chunks)
        chunk_perm = torch.randperm(num_chunks, device=self.device)
        
        for i in chunk_perm:
            chunk = chunks[i]
            
            # 处理 drop_last 逻辑
            if self.drop_last and len(chunk) < self.batch_size:
                continue
            
            yield chunk

    def __len__(self):
        if self.drop_last:
            return self.num_partitions // self.batch_size
        else:
            return (self.num_partitions + self.batch_size - 1) // self.batch_size


def require_module(mod: str, hint: str):
    try:
        return importlib.import_module(mod)
    except ImportError as e:
        raise ImportError(f"Missing optional dependency '{mod}'. {hint}") from e


def _parse_hacs_config(config: Dict[str, Any]) -> Dict[str, Any]:
    hacs_cfg = config.get('HACS')
    if hacs_cfg is None:
        hacs_cfg = config.get('hacs')
    if hacs_cfg is None:
        dps_cfg = config.get('DPS', {})
        hacs_cfg = dps_cfg.get('HACS') or dps_cfg.get('hacs') or {}
    return hacs_cfg or {}


def _compute_hacs_static_scores(features: torch.Tensor,
                                degrees: torch.Tensor,
                                q_min: float,
                                q_max: float,
                                eps: float) -> torch.Tensor:
    feats = features.to(dtype=torch.float32)
    degs = degrees.to(device=feats.device, dtype=torch.float32)
    denom = torch.log(degs + 2.0).clamp(min=1e-6)
    q_raw = feats.norm(p=2, dim=1) / denom
    q_mean = q_raw.mean()
    q_norm = q_raw / (q_mean + eps)
    return torch.clamp(q_norm, q_min, q_max).to(dtype=torch.float32)


def _build_part_ranges(partition_book: list) -> torch.Tensor:
    ranges = []
    for p in partition_book:
        ranges.append([int(p['start_idx']), int(p['end_idx'])])
    if not ranges:
        return torch.empty((0, 2), dtype=torch.int32)
    return torch.tensor(ranges, dtype=torch.int32)


def _build_window_part_ids(part_ranges: torch.Tensor, num_nodes: int, tile_rows: int) -> torch.Tensor:
    num_windows = (int(num_nodes) + tile_rows - 1) // tile_rows
    win_part_ids = torch.full((num_windows,), -1, dtype=torch.int32)
    if part_ranges.numel() == 0:
        return win_part_ids
    for pid, (start, end) in enumerate(part_ranges.tolist()):
        start_win = int(start) // tile_rows
        end_win = int(end) // tile_rows
        win_part_ids[start_win:end_win] = pid
    return win_part_ids


def process_dps_static_graph(config, g_raw, features, labels, train_idx, val_idx, test_idx, device,
                             static_scores: torch.Tensor = None,
                             window_part_ids: torch.Tensor = None,
                             part_ranges: torch.Tensor = None,
                             hacs_params: Dict[str, Any] = None):
    """
    [DPS Index-Based Preprocess] 
    仅运行采样算法生成 Partition Book (Window IDs)，不保存物理子图。
    返回原始的全图结构和索引列表。
    """
    print("  [DPS Preprocess] Running Offline DPS Sampling (Index-Only Mode)...")
    import dgl
    import time
    
    # 1. 初始化参数和全图
    dps_cfg = config.get('DPS', {})
    max_edges_threshold = int(dps_cfg.get('max_partition_edges', 2000000))
    probe_step = int(dps_cfg.get('probe_step_size', 50))
    
    tile_rows = int(config.get('tile_rows', 16))
    tile_cols = int(config.get('tile_cols', 8))

    if hacs_params is None:
        hacs_params = _parse_hacs_config(config)
    hacs_enabled = bool(hacs_params.get('enabled', True))
    alpha = float(hacs_params.get('alpha', 1.0))
    beta = float(hacs_params.get('beta', 10.0))
    delta = float(hacs_params.get('delta', 0.0))
    gamma_bonus = float(hacs_params.get('gamma_bonus', 0.0))
    score_scale = float(hacs_params.get('score_scale', 100.0))
    q_min = float(hacs_params.get('q_min', 0.5))
    q_max = float(hacs_params.get('q_max', 2.0))
    q_eps = float(hacs_params.get('q_eps', 1e-6))

    if hacs_enabled:
        if static_scores is None:
            degs = g_raw.in_degrees()
            if degs.device != features.device:
                degs = degs.to(features.device)
            static_scores = _compute_hacs_static_scores(features, degs, q_min, q_max, q_eps)
        if part_ranges is not None and window_part_ids is None:
            if not isinstance(part_ranges, torch.Tensor):
                part_ranges = torch.tensor(part_ranges, dtype=torch.int32)
            window_part_ids = _build_window_part_ids(part_ranges, g_raw.num_nodes(), tile_rows)
    else:
        static_scores = None
        window_part_ids = None
        part_ranges = None

    locality_enabled = bool(hacs_params.get('locality_enabled', True))
    if not (window_part_ids is not None and part_ranges is not None):
        locality_enabled = False

    # 将原图转换为 BCSR (这就是训练要用的图，常驻显存)
    # 注意：为了预处理采样速度，我们先把图放到 device 上
    bcsr_full = BCSRGraph.from_dgl(g_raw.to(device), tile_rows, tile_cols)
    
    # 2. 初始化采样器
    sampler = BCSRSampler(
        fanouts=config.get('fanouts', [10, 10]),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        warps_per_block=int(config.get('warps_per_block', 8)),
        partition_book=None,
        verbose=False,
        static_scores=static_scores,
        window_part_ids=window_part_ids,
        part_ranges=part_ranges,
        hacs_enabled=hacs_enabled,
        alpha=alpha,
        beta=beta,
        delta=delta,
        gamma_bonus=gamma_bonus,
        score_scale=score_scale,
        locality_enabled=locality_enabled
    )
    
    # 3. 准备种子 Windows
    # 将训练节点转换为 Window ID 并去重
    train_idx_gpu = train_idx.to(device)
    train_window_ids = torch.unique(torch.div(train_idx_gpu, tile_rows, rounding_mode='floor'))
    
    # 打乱顺序
    perm = torch.randperm(train_window_ids.size(0), device=device)
    shuffled_windows = train_window_ids[perm]
    
    # 4. 采样循环
    partition_book = []  # 存储 List[Tensor(WindowIDs)]
    
    total_windows = shuffled_windows.size(0)
    cursor = 0
    part_id = 0
    
    t0 = time.time()
    
    while cursor < total_windows:
        current_valid_seeds = torch.empty(0, dtype=torch.long, device=device)
        accepted_subgraph = None 
        
        # --- 探测阶段 (Probing) ---
        while cursor < total_windows:
            end_ptr = min(cursor + probe_step, total_windows)
            new_seeds = shuffled_windows[cursor : end_ptr]
            temp_seeds = torch.cat([current_valid_seeds, new_seeds])
            
            # 使用在线采样探测子图大小
            # sample_dps_online 内部调用了 repack_subgraph，返回的图是紧凑的
            # temp_bcsr = sampler.sample_dps_online(bcsr_full, temp_seeds)
            temp_bcsr = sampler.sample_dps_online(bcsr_full, temp_seeds, fanouts=sampler.fanouts)
            
            # 计算负载 (边数或 Tile 数)
            if temp_bcsr.values_condensed is not None:
                current_load = temp_bcsr.values_condensed.numel()
            else:
                current_load = temp_bcsr.original_col_indices.numel()

            is_first_step = (current_valid_seeds.size(0) == 0)
            
            # 贪心策略：如果还没超标，就接受；超标了如果是第一步也得硬着头皮接受
            if is_first_step or current_load <= max_edges_threshold:
                accepted_subgraph = temp_bcsr
                current_valid_seeds = temp_seeds
                cursor = end_ptr # 指针推进
                if current_load >= max_edges_threshold: 
                    break # 满了，提交当前分区
            else:
                # 超标了，且不是第一步，丢弃当前增量，回退指针，提交之前的 current_valid_seeds
                del temp_bcsr
                break
        
        if accepted_subgraph is None: 
            break 

        # --- 提交阶段 (Commit) ---
        # 关键修改：我们不再保存 accepted_subgraph 的物理结构 (cols, values)
        # 我们只保存生成它所需的 "原始 Window IDs"
        
        # sampler.py 的 repack_subgraph 会将 original_window_ids 挂载到对象上
        if hasattr(accepted_subgraph, 'original_window_ids'):
            #以此为准，这是 repack 后的有序 Window 集合
            final_windows = accepted_subgraph.original_window_ids.cpu()
        else:
            # 回退策略：使用种子
            final_windows = current_valid_seeds.cpu()

        partition_book.append(final_windows)
        
        # 清理显存
        del accepted_subgraph
        
        part_id += 1
        if part_id % 20 == 0:
            elapsed = time.time() - t0
            print(f"    [DPS-Index] Generated {part_id} partitions. Progress {cursor}/{total_windows}. Time: {elapsed:.1f}s")

    print(f"  [DPS] Done. Generated {len(partition_book)} partitions (Index-Only).")
    
    # 5. 返回
    # 注意：我们返回原始的 bcsr_full，而不是拼接后的巨图
    # train_idx 等保持为全局索引
    # super_map 不再需要，返回 None
    
    return (bcsr_full, features, labels, train_idx, val_idx, test_idx,
            partition_book, None, static_scores, window_part_ids, part_ranges, hacs_enabled)


def apply_rcm_reorder(g, features, labels, train_idx, val_idx, test_idx):
    """
    对图进行 RCM (Reverse Cuthill-McKee) 重排序，以增强数据的内存局部性。
    同时重排特征、标签和数据集切分索引。
    """
    print("  [Preprocess] Applying RCM reordering...")
    dgl = require_module('dgl', "RCM requires dgl")
    
    # 1. 获取邻接矩阵 (Scipy CSR) 并计算 RCM 排列
    # 如果 g 在 GPU 上，需要先转到 CPU，因为 SciPy 仅支持 CPU
    try:
        adj = g.adj_scipy(fmt='csr')
    except (AttributeError, TypeError):
        # 兼容旧版或备用方案：手动构建 CSR
        # 这也是最稳健的方法，不依赖 DGL API 变动
        src, dst = g.edges()
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()
        data = np.ones(len(src))
        import scipy.sparse
        adj = scipy.sparse.csr_matrix((data, (src, dst)), shape=(g.num_nodes(), g.num_nodes()))
        
    perm = scipy.sparse.csgraph.reverse_cuthill_mckee(adj)
    
    # perm 是新图第 i 个节点在原图中的 ID (new_id -> old_id)
    perm_tensor = torch.from_numpy(perm.copy()).long().to(g.device)
    
    # 2. 重排 DGL 图结构
    # dgl.reorder_graph 使用 node_perm 参数重构图
    g_new = dgl.reorder_graph(
        g, 
        node_permute_algo='custom', 
        permute_config={'nodes_perm': perm_tensor}
    )
    
    # 3. 重排节点属性 (Features & Labels)
    # features[i] 应该是原图中 perm[i] 节点的特征
    features_new = features[perm_tensor]
    labels_new = labels[perm_tensor]
    
    # 4. 映射切分索引 (Train/Val/Test Indices)
    # 原始索引是 old_id，我们需要找到它们对应的 new_id
    # 构建反向映射: old_id -> new_id
    inv_perm = torch.zeros_like(perm_tensor)
    inv_perm[perm_tensor] = torch.arange(len(perm_tensor), device=g.device, dtype=torch.long)
    
    train_idx_new = inv_perm[train_idx]
    val_idx_new = inv_perm[val_idx]
    test_idx_new = inv_perm[test_idx]
    
    return g_new, features_new, labels_new, train_idx_new, val_idx_new, test_idx_new


def apply_baps_reorder(g, features, labels, train_idx, val_idx, test_idx, num_partitions, tile_rows, use_intra_rcm=True):
    """
    BAPS 核心预处理: METIS 分区 + [可选]分区内 RCM + Window 对齐 Padding
    """
    if use_intra_rcm:
        print(f"  [BAPS Preprocess] Running METIS (k={num_partitions}) + Intra-Partition RCM + Padding...")
    else:
        print(f"  [BAPS Preprocess] Running METIS (k={num_partitions}) + Padding (No RCM)...")
    
    # 依赖检查
    dgl = require_module('dgl', "BAPS requires dgl")
    import scipy.sparse as sp
    
    # 1. 宏观划分 (METIS)
    # 确保图在 CPU 上进行 METIS
    g_cpu = g.cpu()
    # 如果图没有边权重，METIS 可能会报错或效果不好，通常不需要额外权重
    partition_ids = dgl.metis_partition_assignment(g_cpu, num_partitions)
    
    # 准备构建新的对齐图
    new_node_pointer = 0
    node_mapping = torch.full((g.num_nodes(),), -1, dtype=torch.long) # Old ID -> New ID
    
    # 收集分区的 CSR 数据以构建新图
    row_list = []
    col_list = []
    data_list = []
    
    # 获取原始图的邻接矩阵 (CSR)
    try:
        adj = g_cpu.adj_external(scipy_fmt='csr')
    except AttributeError:
        # 兼容旧版本 DGL
        adj = g_cpu.adj(scipy_fmt='csr')
    
    # 记录分区元数据 (用于 PartitionDataLoader)
    # 格式: list of (start_node_id, end_node_id, actual_count, padded_count)
    partition_book = [] 
    
    # 用于重排特征和标签
    indices_for_feature_copy = [] # 存储 (new_idx, old_idx) 用于最后复制
    
    # 2. 遍历每个分区进行处理
    for pid in range(num_partitions):
        # 2.1 提取该分区的节点
        mask = (partition_ids == pid)
        nodes_in_part = torch.nonzero(mask, as_tuple=True)[0].numpy()
        num_nodes_part = len(nodes_in_part)
        
        if num_nodes_part == 0:
            continue
            
        # 2.2 分区内微观排序 (Intra-Partition RCM)
        if use_intra_rcm:
            # 提取子矩阵并计算 RCM
            sub_adj = adj[nodes_in_part, :][:, nodes_in_part]
            perm_local = sp.csgraph.reverse_cuthill_mckee(sub_adj)
            # 获取排序后的原始节点 ID
            nodes_sorted = nodes_in_part[perm_local]
        else:
            # 纯 METIS 模式：保持节点在原图中的相对顺序 (或者随机，取决于 METIS 输出，通常是 ID 升序)
            nodes_sorted = nodes_in_part
        
        # 2.3 计算对齐 Padding
        # 我们需要凑成 tile_rows (16) 的整数倍
        remainder = num_nodes_part % tile_rows
        pad_count = (tile_rows - remainder) % tile_rows
        total_count = num_nodes_part + pad_count
        
        # 记录映射关系 (Old -> New)
        # 当前分区的起始 New ID 是 new_node_pointer
        current_new_ids = np.arange(new_node_pointer, new_node_pointer + num_nodes_part)
        node_mapping[nodes_sorted] = torch.from_numpy(current_new_ids)
        
        # 记录特征复制的索引对
        # 我们稍后会创建一个全零的大矩阵，只把真实数据填进去
        # 这里记录：features_new[current_new_ids] = features[nodes_sorted]
        indices_for_feature_copy.append((current_new_ids, nodes_sorted))
        
        # 2.4 构建新图的边 (需要重映射列索引)
        # 这一步比较 tricky：边的源节点已经按分区块排好了，但目标节点(列)的 ID 还是旧的
        # 我们暂时只收集旧的边，等所有节点的 mapping 建立好后，统一转换列索引
        
        # 取出排序后的节点对应的行
        # 注意：这里取的是全图 adj 的行，所以列索引还是 Old Global ID
        sub_adj_sorted = adj[nodes_sorted, :]
        row_idx, col_idx = sub_adj_sorted.nonzero()
        
        # 行索引需要平移到新图的位置
        # sub_adj_sorted 的第 i 行 对应新图的 new_node_pointer + i 行
        row_idx_shifted = row_idx + new_node_pointer
        
        row_list.append(row_idx_shifted)
        col_list.append(col_idx) # 暂时保持 Old ID
        data_list.append(sub_adj_sorted.data)
        
        # 记录元数据
        partition_book.append({
            'id': pid,
            'start_idx': new_node_pointer,
            'end_idx': new_node_pointer + total_count, # 包含 padding
            'real_count': num_nodes_part,
            'pad_count': pad_count
        })
        
        # 更新指针 (跳过 Padding 区域，Dummy 节点没有边)
        new_node_pointer += total_count

    # 3. 构建包含 Padding 的全局新图
    final_num_nodes = new_node_pointer
    
    # 合并边列表
    all_rows = np.concatenate(row_list)
    all_cols = np.concatenate(col_list) # 这里还是 Old ID
    all_data = np.concatenate(data_list)
    
    # [关键] 将列索引 (Old ID) 转换为 New ID
    # 注意：如果原图中有节点因为某种原因没被划分到任何分区(不太可能)，这里会出问题，需确保覆盖率
    # node_mapping 是 torch tensor，转 numpy 查表
    all_cols_new = node_mapping[all_cols].numpy()
    
    # 过滤掉无效边 (如果源节点指向了被丢弃的节点，虽然理论上 METIS 覆盖全图)
    valid_mask = all_cols_new != -1

    # 定义 new_adj (这一步会自动合并重复边)
    new_adj = sp.csr_matrix(
        (all_data[valid_mask], (all_rows[valid_mask], all_cols_new[valid_mask])),
        shape=(final_num_nodes, final_num_nodes)
    )
    
    # 转回 DGL 图
    g_new = dgl.from_scipy(new_adj)

    # 显式将权重保存到 edata，确保后续转 BCSR 时能带上
    # 注意：float32 足够，float64 浪费
    # g_new.edata['weight'] = torch.from_numpy(new_adj.data).float()
    
    # 4. 重构特征和标签 (包含 Zero-Padding)
    # 创建全零/空特征
    feat_dim = features.shape[1]
    features_new = torch.zeros((final_num_nodes, feat_dim), dtype=features.dtype, device=features.device)
    
    # [修改] 根据原始标签维度自动初始化 (适配 PPI/Yelp 等多标签数据)
    if labels.dim() > 1:
        # 多标签情况: (N, NumClasses)
        label_dim = labels.shape[1]
        # 使用 -1 (或 0) 填充，视具体 dtype 而定，通常 float 用 0 或 -1 均可，这里保持一致
        labels_new = torch.full((final_num_nodes, label_dim), -1, dtype=labels.dtype, device=labels.device)
    else:
        # 单标签情况: (N,)
        labels_new = torch.full((final_num_nodes,), -1, dtype=labels.dtype, device=labels.device)
    
    # 填充真实数据
    for new_ids, old_ids in indices_for_feature_copy:
        # 确保 old_ids 在 device 上
        old_ids_tensor = torch.from_numpy(old_ids).to(features.device)
        new_ids_tensor = torch.from_numpy(new_ids).to(features.device)
        
        features_new[new_ids_tensor] = features[old_ids_tensor]
        labels_new[new_ids_tensor] = labels[old_ids_tensor]

    # 5. 映射切分索引 (Train/Val/Test)
    def map_indices(old_idx):
        new_idx = node_mapping[old_idx.cpu()]
        return new_idx[new_idx != -1].to(train_idx.device) # 保持原设备
        
    train_idx_new = map_indices(train_idx)
    val_idx_new = map_indices(val_idx)
    test_idx_new = map_indices(test_idx)
    
    # 将 partition_book 保存到 g_new 中，或者作为 DataBundle 的一部分返回
    # 为了简单，我们将其挂载到 g_new.edata 或 ndata 是存不下的，直接返回即可
    
    return g_new, features_new, labels_new, train_idx_new, val_idx_new, test_idx_new, partition_book


class BaseAdapter:
    def load(self, ds_cfg: Dict[str, Any], tile_rows: int, tile_cols: int, device, apply_rcm: bool = False) -> Tuple[DataBundle, Any]:
        raise NotImplementedError


class DGLAdapter(BaseAdapter):
    """通用 DGL 数据集适配器（支持通过 module/class/kwargs 指定任意 DGL 内置数据集）。"""
    def load(self, ds_cfg: Dict[str, Any], tile_rows: int, tile_cols: int, device, apply_rcm: bool = False) -> DataBundle:
        dgl = require_module('dgl', "Install via: pip install dgl (or dgl-cuXX for CUDA)")
        
        name = ds_cfg.get('name')
        cls_name = ds_cfg.get('class')
        
        # --- [新增] 1. 特殊数据集处理逻辑 ---
        if name == 'ppi':
            print("[Data] Detected PPI dataset. Merging Train/Val/Test graphs...")
            # 分别加载三个部分
            train_ds = dgl.data.PPIDataset(mode='train')
            valid_ds = dgl.data.PPIDataset(mode='valid')
            test_ds = dgl.data.PPIDataset(mode='test')
            
            # 合并所有子图 (20 train + 2 val + 2 test)
            g_list = list(train_ds) + list(valid_ds) + list(test_ds)
            g = dgl.batch(g_list)
            
            # 手动生成 Mask
            batch_num_nodes = g.batch_num_nodes()
            n_train = sum(batch_num_nodes[:20])
            n_val = sum(batch_num_nodes[20:22])
            # n_test = sum(batch_num_nodes[22:])
            
            train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
            val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
            test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train : n_train + n_val] = True
            test_mask[n_train + n_val :] = True
            
            g.ndata['train_mask'] = train_mask
            g.ndata['val_mask'] = val_mask
            g.ndata['test_mask'] = test_mask
            
            feature_key, label_key = 'feat', 'label'

        elif name == 'yelp':
            print("[Data] Detected Yelp dataset.")
            ds = dgl.data.YelpDataset()
            g = ds[0]
            feature_key, label_key = 'feat', 'label'
            
        else:
            # 通用逻辑（Amazon 等）
            module = ds_cfg.get('module', 'dgl.data')
            
            # 简单映射 Amazon
            if not cls_name and name == 'amazon_computer':
                cls_name = 'AmazonCoBuyComputerDataset'
                
            if not cls_name:
                raise ValueError(f"For source=dgl, please provide dataset.class in config for {name}.")
            
            kwargs = ds_cfg.get('kwargs', {})
            feature_key = ds_cfg.get('feature_key', 'feat')
            label_key = ds_cfg.get('label_key', 'label')

            mod = importlib.import_module(module)
            ds_cls = getattr(mod, cls_name)
            ds = ds_cls(**kwargs)
            g = ds[0]

        feats = g.ndata[feature_key]
        labels = g.ndata[label_key]

        # --- [新增] 2. 检查并生成 Mask (针对 Amazon 等无默认划分的数据集) ---
        train_mask_key = ds_cfg.get('train_mask_key', 'train_mask')
        val_mask_key = ds_cfg.get('val_mask_key', 'val_mask')
        test_mask_key = ds_cfg.get('test_mask_key', 'test_mask')

        if train_mask_key in g.ndata:
            train_idx = torch.nonzero(g.ndata[train_mask_key], as_tuple=True)[0]
            val_idx = torch.nonzero(g.ndata[val_mask_key], as_tuple=True)[0]
            test_idx = torch.nonzero(g.ndata[test_mask_key], as_tuple=True)[0]
        else:
            print("[Data] Masks not found. Generating random splits (60/20/20)...")
            n_nodes = g.num_nodes()
            indices = torch.randperm(n_nodes)
            n_train = int(n_nodes * 0.6)
            n_val = int(n_nodes * 0.2)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

        # 图结构预处理
        g = dgl.to_bidirected(g)
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        
        # 归一化权重计算
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        src, dst = g.edges()
        norm = norm.to(src.device) # 确保设备一致
        edge_weights = norm[src] * norm[dst]
        g.edata['weight'] = edge_weights

        # RCM 重排序
        if apply_rcm:
            g, feats, labels, train_idx, val_idx, test_idx = apply_rcm_reorder(
                g, feats, labels, train_idx, val_idx, test_idx
            )

        g = g.to(device)
        bcsr = BCSRGraph.from_dgl(g, tile_rows=tile_rows, tile_cols=tile_cols)

        # --- [新增] 3. 标签维度自适应处理 (Multilabel 支持) ---
        feats = feats.to(device, non_blocking=True).float()
        
        # 如果标签是 2D 且第二维 > 1，则视为多标签分类 (PPI/Yelp)
        if labels.dim() > 1 and labels.shape[1] > 1:
            # 多标签需要 Float 类型以配合 BCEWithLogitsLoss
            labels = labels.to(device, non_blocking=True).float()
            num_classes = labels.shape[1]
        else:
            # 单标签 (Cora/Amazon)，转为 1D Long
            labels = labels.to(device, non_blocking=True).long().view(-1)
            num_classes = int(labels.max().item() + 1)

        bundle = DataBundle(
            bcsr,
            feats,
            labels,
            train_idx.to(device),
            val_idx.to(device),
            test_idx.to(device),
            num_classes,
            partition_book=None
        )

        # 返回 (bundle, g) 元组，以便外部使用 g 进行 BAPS 处理
        return bundle, g


class OGBAdapter(BaseAdapter):
    """OGB 的 DGL 版节点属性预测数据集适配器（通用 ogbn-*）。"""
    def load(self, ds_cfg: Dict[str, Any], tile_rows: int, tile_cols: int, device, apply_rcm: bool = False) -> DataBundle:
        nodeprop = require_module('ogb.nodeproppred', "Install via: pip install ogb")
        dgl = require_module('dgl', "Install via: pip install dgl (or dgl-cuXX for CUDA)")
        DglNodePropPredDataset = getattr(nodeprop, 'DglNodePropPredDataset')
        name = ds_cfg.get('name')
        if not name:
            raise ValueError("For source=ogb, please provide dataset.name (e.g., ogbn-products)")
        ds = DglNodePropPredDataset(name=name)
        g, labels = ds[0]
        split_idx = ds.get_idx_split()

        # [修复] 1. 提前提取特征！
        # 在进行 to_bidirected 等破坏性操作前，先拿到特征的引用
        # 这样无论后续图结构怎么变，特征数据都在我们手里
        feat = g.ndata['feat']

        # [改动] 2. 转为无向图
        g = dgl.to_bidirected(g)

        # 这一步会去除重复的 u->v 边，只保留一条
        g = dgl.to_simple(g)

        # [改动] 3. 添加自环
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        # [改动] 4. 计算对称归一化权重 (Symmetric Normalization)
        # 计算对称归一化权重 (Symmetric Normalization)
        deg = g.in_degrees().float().clamp(min=1)  # 获取入度
        norm = torch.pow(deg, -0.5)                # 计算 D^{-1/2}
        src, dst = g.edges()
        norm = norm.to(src.device)
        edge_weights = norm[src] * norm[dst]       # 计算 D^{-1/2} * A * D^{-1/2}
        g.edata['weight'] = edge_weights           # 存入图数据中

        # RCM 重排序
        # 注意：这里我们要传入 feat (之前提取的特征引用) 和 split_idx 中的 tensor
        if apply_rcm:
            g, feat, labels, train_idx, val_idx, test_idx = apply_rcm_reorder(
                g, feat, labels, split_idx['train'], split_idx['valid'], split_idx['test']
            )
            split_idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}

        # BCSR 转换 (保持在 CPU)
        bcsr = BCSRGraph.from_dgl(g, tile_rows=tile_rows, tile_cols=tile_cols)
        bcsr = bcsr.to(device)

        # [修复] 5. 使用第一步提取的 feat，而不是 g.ndata['feat']
        feats = feat.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long().view(-1)

        num_classes = int(labels.max().item() + 1)
        
        bundle = DataBundle(
            bcsr,
            feats,
            labels,
            split_idx['train'].to(device),
            split_idx['valid'].to(device),
            split_idx['test'].to(device),
            num_classes,
            partition_book=None
        )

        return bundle, g


def build_adapter(ds_cfg: Dict[str, Any]) -> BaseAdapter:
    source = (ds_cfg or {}).get('source')
    if source == 'dgl':
        return DGLAdapter()
    if source == 'ogb':
        return OGBAdapter()
    raise ValueError("dataset.source must be one of: dgl, ogb")


def _add_norm_weights(g, device):
    g = g.to(device) # 确保计算在 GPU (如果需要)
    deg = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    src, dst = g.edges()
    # 确保 norm 和 indices 在同一设备
    norm = norm.to(src.device)
    edge_weights = norm[src] * norm[dst]
    g.edata['weight'] = edge_weights
    return g


def load_dataset(config: Dict[str, Any], device) -> DataBundle:
    # 1. 基础参数读取
    tile_rows = int(config.get('tile_rows', 16))
    tile_cols = int(config.get('tile_cols', 8))
    ds_cfg = config.get('dataset', {})

    # [读取训练模式]
    train_mode = config.get('train_mode', 'node')  # 默认为 node 模式

    # [读取预处理模式]
    cfg_preprocess = config.get('preprocess_mode', None)

    # --- 策略解析逻辑 ---
    if train_mode == 'BAPS':
        # BAPS 强制依赖 METIS
        preprocess_mode = 'metis+rcm'
        if cfg_preprocess and cfg_preprocess.lower() != 'metis':
            print(f"[Data] Warning: train_mode='BAPS' requires METIS. Overriding preprocess_mode '{cfg_preprocess}' to 'metis'.")
    elif train_mode == 'node':
        # node 模式默认 RCM
        preprocess_mode = cfg_preprocess if cfg_preprocess else 'rcm'
    elif train_mode == 'DPS':
        # DPS 模式默认 none (原始顺序)，但也常配合 RCM/METIS
        preprocess_mode = cfg_preprocess if cfg_preprocess else 'none'
    else: # full 等其他模式
        preprocess_mode = cfg_preprocess if cfg_preprocess else 'rcm'

    preprocess_mode = preprocess_mode.lower()
    print(f"[Data] Train Mode: {train_mode} | Preprocess Mode: {preprocess_mode}")

    use_rcm = False
    use_metis = False
    use_intra_rcm = False
    num_partitions = 100 # 默认值

    if preprocess_mode == 'rcm':
        use_rcm = True
    elif preprocess_mode == 'metis':
        use_metis = True
        use_intra_rcm = False # [明确] 纯 METIS，不做内部 RCM
        metis_cfg = config.get('metis', {})
        if not metis_cfg: metis_cfg = config.get('BAPS', {})
        num_partitions = int(metis_cfg.get('num_partitions', 100))
    elif preprocess_mode == 'metis+rcm':
        use_metis = True
        use_intra_rcm = True  # [明确] METIS + 内部 RCM
        metis_cfg = config.get('metis', {})
        if not metis_cfg: metis_cfg = config.get('BAPS', {})
        num_partitions = int(metis_cfg.get('num_partitions', 100))
    elif preprocess_mode == 'none':
        pass
    else:
        raise ValueError(f"Unknown preprocess_mode: {preprocess_mode}. Use 'rcm', 'metis', 'metis+rcm', or 'none'.")

    hacs_params = None
    hacs_enabled = False
    if train_mode == 'DPS':
        hacs_params = _parse_hacs_config(config)
        hacs_enabled = bool(hacs_params.get('enabled', True))
    
    adapter = build_adapter(ds_cfg)
    use_uva = config.get('use_uva', False)
    if use_uva or (train_mode == 'DPS'):
        load_device = torch.device('cpu')
    else:
        load_device = device

    # 2. 生成缓存路径与文件名
    ds_name = ds_cfg.get('name')
    if not ds_name and ds_cfg.get('source') == 'dgl':
        ds_name = ds_cfg.get('class', 'default_dgl')
    if not ds_name:
        ds_name = 'default'

    # 构造文件名后缀
    if use_metis:
        if use_intra_rcm:
            prep_suffix = f"_metis_rcm_p{num_partitions}" # 新后缀
        else:
            prep_suffix = f"_metis_p{num_partitions}"     # 纯 METIS 后缀
    elif use_rcm:
        prep_suffix = "_rcm"
    else:
        prep_suffix = "_raw"

    # 文件命名逻辑 (根据 train_mode 和关键参数)
    if train_mode == 'DPS':
        dps_cfg = config.get('DPS', {})
        max_edges = int(dps_cfg.get('max_partition_edges', 2000000))
        step_size = int(dps_cfg.get('probe_step_size', 50))
        strategy = dps_cfg.get('strategy', 'seed_increment')
        storage_format = f"_{train_mode}_edge{max_edges}_step{step_size}_{strategy}"
        if hacs_enabled:
            storage_format = f"{storage_format}_hacs"
    elif train_mode == 'BAPS':
        storage_format = f"_{train_mode}_p{num_partitions}"
    elif train_mode == 'node':
        storage_format = f"_{train_mode}"
    elif train_mode == 'full':
        storage_format = f"_{train_mode}"
    else:
        storage_format = f"_unknown"

    cache_filename = f"{ds_name}{prep_suffix}{storage_format}.pt"

    root_dir = ds_cfg.get('root', './dataset')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    cache_path = os.path.join(root_dir, cache_filename)
    partition_book_path = cache_path.replace('.pt', '_partition_book.pt')

    # 3. 加载或处理逻辑
    partition_book = None
    
    if os.path.exists(cache_path):
        print(f"[Data] Loading cached dataset from: {cache_path}")
        bundle = torch.load(cache_path, weights_only=False)
        
        # 尝试加载 partition_book (如果是 DPS 或 BAPS/METIS 生成的)
        if os.path.exists(partition_book_path):
            partition_book = torch.load(partition_book_path, weights_only=False)
            bundle = bundle._replace(partition_book=partition_book)
            
    else:
        print(f"[Data] Cache not found. Processing dataset (Mode={train_mode}, Preprocess={preprocess_mode})...")
        
        # [Step 1] 初始加载
        # 如果要用 METIS，初始加载时不需要 RCM (METIS 自己会处理内部排序)
        # 如果只要 RCM，则让 adapter 直接做
        adapter_apply_rcm = use_rcm and (not use_metis)
        bundle, g_curr = adapter.load(ds_cfg, tile_rows, tile_cols, load_device, apply_rcm=adapter_apply_rcm)
        
        # [Step 2] 统一处理 METIS 重排序 (解耦核心：无论什么 Train Mode，只要配了 METIS 都在这里执行)
        if use_metis:
            # 执行 METIS 分区 + 内部 RCM + Padding 对齐
            # 这会改变 g_curr 的结构，以及 features/labels 的顺序
            g_curr, f_new, l_new, tr_new, va_new, te_new, metis_book = apply_baps_reorder(
                g_curr, bundle.features, bundle.labels, 
                bundle.train_idx, bundle.val_idx, bundle.test_idx,
                num_partitions, tile_rows,
                use_intra_rcm=use_intra_rcm
            )

            g_curr = _add_norm_weights(g_curr, device)
            
            # 将重排后的图转为 BCSR
            # 注意：apply_baps_reorder 返回的 g_curr 是 DGL 图
            g_curr = g_curr.to(device)
            bcsr_new = BCSRGraph.from_dgl(g_curr, tile_rows=tile_rows, tile_cols=tile_cols)
            
            # 更新 bundle
            # 注意：这里的 partition_book 是 METIS 的物理分区表
            bundle = DataBundle(
                bcsr_new, f_new, l_new, 
                tr_new, va_new, te_new, 
                bundle.num_classes, 
                partition_book=metis_book
            )
            
            # 如果是 BAPS 模式，这个 partition_book 就是最终要用的
            if train_mode == 'BAPS':
                partition_book = metis_book
            
            mode_str = "METIS + Intra-RCM" if use_intra_rcm else "METIS (No RCM)"
            print(f"  [Data] {mode_str} preprocessing done. Graph reordered.")

        static_scores = None
        window_part_ids = None
        part_ranges = None

        if train_mode == 'DPS' and hacs_enabled and use_metis:
            part_ranges = _build_part_ranges(metis_book)
            window_part_ids = _build_window_part_ids(part_ranges, g_curr.num_nodes(), tile_rows)

        # [Step 3] 处理 DPS 模式特有的逻辑
        # DPS 需要在当前图（可能是 Raw, RCM, 或 METIS 后的图）基础上生成 Window Partition Book
        if train_mode == 'DPS':
            # 注意：process_dps_static_graph 内部也会把 g_curr 转 BCSR，但我们需要传入当前的 bundle 数据
            # 为了避免重复转换 BCSR，我们可以稍微优化 process_dps_static_graph，
            # 但为了保持接口兼容，这里我们传入 g_curr。
            # 如果 Step 2 跑了 METIS，g_curr 已经是重排后的图；如果没跑，就是 adapter 返回的图。
            
            (bcsr_final, f_final, l_final, tr_final, va_final, te_final,
             dps_book, super_map, static_scores, window_part_ids, part_ranges,
             hacs_enabled) = process_dps_static_graph(
                config, g_curr, bundle.features, bundle.labels,
                bundle.train_idx, bundle.val_idx, bundle.test_idx, device,
                static_scores=static_scores,
                window_part_ids=window_part_ids,
                part_ranges=part_ranges,
                hacs_params=hacs_params
            )
            
            partition_book = dps_book
            bundle = DataBundle(
                bcsr_final, f_final, l_final, 
                tr_final, va_final, te_final,
                bundle.num_classes, partition_book=partition_book,
                super_map=super_map,
                static_scores=static_scores,
                window_part_ids=window_part_ids,
                part_ranges=part_ranges,
                hacs_enabled=hacs_enabled
            )

        # [Step 4] BAPS 模式检查
        elif train_mode == 'BAPS':
            # BAPS 模式直接使用 Step 2 生成的数据和 partition_book
            if partition_book is None:
                raise ValueError("[Data] BAPS mode requires METIS preprocessing. Something went wrong.")
        
        # [Step 5] Node / Full 模式
        else:
            # 此时 bundle 已经是正确排序（Raw / RCM / METIS）的数据了
            # 不需要生成额外的 partition_book
            pass

        # 保存缓存
        torch.save(bundle, cache_path)
        if partition_book is not None:
            torch.save(partition_book, partition_book_path)

   # 4. 后续通用处理 (特征归一化、Padding、UVA 等)
    features = bundle.features
    labels = bundle.labels
    bcsr = bundle.bcsr_full

    # 安全特征行归一化 (Safe Normalization)
    # 防止全零向量导致 NaN (F.normalize 可能在 norm=0 时产生问题)
    if config.get('normalize_features', True):
        print("[Data] Applying Safe Feature Normalization...")
        # 计算 L2 范数
        norms = features.norm(p=2, dim=1, keepdim=True)
        # 将范数限制在最小 1e-12，避免除以 0
        norms.clamp_(min=1e-12)
        features = features / norms
        
        bundle = bundle._replace(features=features)
        
    # Padding Check
    # 无论是否 DPS 模式，特征矩阵都必须与 BCSR 图结构的物理尺寸对齐
    expected_rows = bcsr.num_windows * bcsr.tile_rows
    if features.shape[0] < expected_rows:
        print(f"[Data] Padding features/labels to match BCSR alignment: {features.shape[0]} -> {expected_rows}")
        pad_len = expected_rows - features.shape[0]
        
        # 对 Feature 进行 Padding (填 0)
        features = F.pad(features, (0, 0, 0, pad_len), "constant", 0)
        
        # 对 Label 进行 Padding (填 0 而不是 -1)
        # 注意：这些 Padding 节点的 Mask 都是 False，不会参与 Loss 计算。
        # 但填 0 可以防止万一泄露导致的 CUDA 错误/NaN。
        if labels.dim() > 1:
            labels = F.pad(labels, (0, 0, 0, pad_len), "constant", 0)
        else:
            labels = F.pad(labels, (0, pad_len), "constant", 0)
    static_scores = getattr(bundle, 'static_scores', None)
    if static_scores is not None and static_scores.shape[0] < expected_rows:
        pad_len = expected_rows - static_scores.shape[0]
        static_scores = F.pad(static_scores, (0, pad_len), "constant", 0)
        bundle = bundle._replace(static_scores=static_scores)

    # UVA 优化逻辑
    # 策略: DPS 模式下生成的特征极大，默认保留在 CPU
    # 逻辑：即使是 DPS 模式，如果显存够，也最好放 GPU
    # 我们可以通过 config.get('gpu_cache_features', False) 来控制
    gpu_cache_features = config.get('gpu_cache_features', False) 
    
    # 修改判断逻辑：
    keep_feats_on_cpu = (train_mode == 'DPS') and (not use_uva) and (not gpu_cache_features)
    
    if use_uva:
        # 特征必须在 CPU 且 Pin 住，以便 GPU 通过 PCIe 快速读取
        if not features.is_cuda:
            features = features.share_memory_().pin_memory()
        
        # [Fix] Pin BCSR structures ONLY if they are on CPU
        # 如果是 DPS 模式，bcsr 可能已经在 GPU 上了，此时不能 pin
        if hasattr(bcsr, 'window_offset'): 
            if not bcsr.window_offset.is_cuda:
                bcsr.window_offset = bcsr.window_offset.pin_memory()

        if hasattr(bcsr, 'original_col_indices'): 
            if not bcsr.original_col_indices.is_cuda:
                bcsr.original_col_indices = bcsr.original_col_indices.pin_memory()

        if hasattr(bcsr, 'values_condensed') and bcsr.values_condensed is not None: 
            if not bcsr.values_condensed.is_cuda:
                bcsr.values_condensed = bcsr.values_condensed.pin_memory()
        
        # Labels 通常很小，直接放 GPU
        if not labels.is_cuda:
            labels = labels.to(device, non_blocking=True) 
            
    else:
        # Non-UVA
        if not keep_feats_on_cpu:
            if not features.is_cuda and device.type == 'cuda': features = features.to(device)
            if not labels.is_cuda and device.type == 'cuda': labels = labels.to(device)
        else:
            # Keep on CPU, but make sure Graph Structure is on GPU if needed
            if not bcsr.window_offset.is_cuda and device.type == 'cuda':
                bcsr = bcsr.to(device)
            # Labels usually fit in GPU
            if not labels.is_cuda and device.type == 'cuda': 
                labels = labels.to(device)
        
    return DataBundle(
        bcsr, 
        features, 
        labels, 
        bundle.train_idx, 
        bundle.val_idx, 
        bundle.test_idx, 
        bundle.num_classes, 
        bundle.partition_book,
        getattr(bundle, 'super_map', None),
        getattr(bundle, 'static_scores', None),
        getattr(bundle, 'window_part_ids', None),
        getattr(bundle, 'part_ranges', None),
        getattr(bundle, 'hacs_enabled', False)
    )
