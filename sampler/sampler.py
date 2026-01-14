# Graph_RWS/sampler/sampler.py
"""
纯 BCSR 版本的自定义图采样器（无 DGL 依赖）
- 直接基于 BCSRGraph 与自定义 CUDA 采样核进行多层采样
- 返回每层的 BCSR 子图列表（从第0层到第L-1层，顺序与模型层一致）

使用方法：
    sampler = BCSRSampler(fanouts=[10, 10], tile_rows=16, tile_cols=8)
    layers_bcsr = sampler.sample_layers(bcsr_full_graph, seed_nodes)
    # 然后直接传入模型： model(layers_bcsr, features)
"""

from typing import List
import torch
import numpy as np

from ..data import BCSRGraph

# 导入 CUDA 采样函数
try:
    from ..core import sample_all_layers
except ImportError:
    print("=" * 60)
    print("WARNING: CUDA sampling kernels not found!")
    print("Please compile the core extension first:")
    print("  cd core && python setup.py build_ext --inplace")
    print("=" * 60)
    sample_all_layers = None


class BCSRSampler:
    """
    纯 BCSR 多层采样器（无 DGL）。

    - 输入：完整图（BCSRGraph）和第一层的目标节点 seed_nodes（全局节点 ID，int64/int32）。
    - 内部：将目标节点映射为“行 window”（以 tile_rows 为步长），
            调用 CUDA 采样核针对每一层做 window 级别的采样，返回 BCSR 子图。
    - 输出：List[BCSRGraph]，长度等于 fanouts 层数，顺序与模型层一致（从浅到深）。
    """

    def __init__(self,
                 fanouts: List[int],
                 tile_rows: int = 16,
                 tile_cols: int = 8,
                 warps_per_block: int = 8,
                 partition_book: list = None,
                 verbose: bool = True):
        self.fanouts = fanouts
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.warps_per_block = warps_per_block
        self.partition_book = partition_book
        self.verbose = verbose
        self.time_tracker = {}

    # ---------- 内部工具 ----------
    @staticmethod
    def _to_cuda_int(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.device.type != 'cuda':
            x = x.to('cuda', non_blocking=True)
        if x.dtype != torch.int32:
            x = x.to(torch.int32)
        return x

    def _build_subgraph(self,
                        out_win_off: torch.Tensor,
                        out_col_idx: torch.Tensor,
                        out_values: torch.Tensor,
                        num_cols: int,
                        num_windows: int) -> BCSRGraph:
        """根据采样核输出的组件构造一个 BCSRGraph 子图。"""
        num_rows = int(num_windows) * self.tile_rows
        return BCSRGraph.from_components(
            window_offset=out_win_off,
            original_col_indices=out_col_idx,
            values_condensed=out_values,
            num_rows=num_rows,
            num_cols=int(num_cols),
            tile_rows=self.tile_rows,
            tile_cols=self.tile_cols,
            device=out_win_off.device
        )

    def _print_performance_summary(self, timing, layer_idx):
        """打印性能摘要"""
        print(f"\n------------ [Layer {layer_idx}] PERFORMANCE SUMMARY ------------")
        print(f"    [Sampling CUDA] Detailed Breakdown:")
        print(f"      Phase 1 - Column Sampling:{timing.phase1_sampling_time:10.3f} ms")
        print(f"      Phase 2 - Value Mapping  :{timing.phase2_mapping_time:10.3f} ms")
        print(f"      Phase 3 - Recompression  :{timing.phase3_recompression_time:10.3f} ms")
        print(f"    -----------------------------------------")
        print(f"      Total CUDA Time:          {timing.total_time:10.3f} ms")
        print("-------------------------------------------------------")

    # ---------- 对外接口 ----------
    def sample_layers(self, g: BCSRGraph, seed_nodes: torch.Tensor) -> List[BCSRGraph]:
        """
        执行多层采样。

        Args:
            g: 完整图（BCSRGraph）
            seed_nodes: 第0层目标节点
        Returns:
            List[BCSRGraph]: 每层的子图列表（顺序与网络层一致）。
        """
        # 1. 语义映射：将“单节点 Fanout”转换为“Window Fanout”
        effective_fanouts = []
        for f in self.fanouts:
            window_budget = f * self.tile_rows
            effective_fanouts.append(window_budget)

        # 模型层顺序是 [L0, L1, L2] (输入->输出)
        # 采样顺序是 [L2, L1, L0] (输出->输入，从 Seed 节点向外扩)
        reversed_fanouts = list(reversed(self.fanouts))
        seed_nodes = self._to_cuda_int(seed_nodes)
        
        # [Fix 1] 核心修复：将 window_offset 转换为 int32
        # BCSRGraph 默认存储 int64，但 C++ Kernel 期望 int32 指针
        # 如果不转，Kernel 会读取错误的 stride，导致偏移量错乱，引发 Illegal Memory Access。
        win_offset_i32 = g.window_offset.to(torch.int32)
        
        # 2. 调用 C++ 接口
        all_layers_data, timing_list = sample_all_layers(
            g.window_offset,          # 使用转换后的 tensor
            g.original_col_indices,
            g.values_condensed,
            seed_nodes,
            reversed_fanouts,
            int(g.num_nodes),
            self.tile_rows,
            self.tile_cols,
            self.warps_per_block
        )

        # 3. 详细时间统计
        if self.verbose:
            num_layers = len(self.fanouts)
            for i, timing in enumerate(timing_list):
                layer_id = (num_layers - 1) - i
                
                if layer_id not in self.time_tracker:
                    self.time_tracker[layer_id] = {
                        'count': 0, 'p1_sample': 0.0, 'p2_map': 0.0, 'p3_recomp': 0.0, 'total': 0.0
                    }
                
                stats = self.time_tracker[layer_id]
                stats['count'] += 1
                stats['p1_sample']   += timing.phase1_sampling_time
                stats['p2_map']      += timing.phase2_mapping_time
                stats['p3_recomp']   += timing.phase3_recompression_time
                stats['total']       += timing.total_time

        # 4. 结果重组
        layers: List[BCSRGraph] = []

        for i, tensors in enumerate(all_layers_data):
            fwd_win, fwd_col, fwd_val, active_win, active_mask = tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]
                        
            # 直接构造标准的 BCSRGraph
            sub_graph = self._build_subgraph(
                fwd_win, fwd_col, fwd_val, 
                num_cols=g.num_cols, 
                num_windows=active_win.size(0)
            )

            # 将 active_windows 挂载到 sub_graph 上
            # model.py 的 Reindex 需要用到这个属性
            sub_graph.active_windows = active_win
            sub_graph.active_masks = active_mask
            
            # 插入到列表头部 (保持 Input -> Output 顺序)
            layers.insert(0, sub_graph)

        return layers

    def __repr__(self):
        return (f"BCSRSampler(fanouts={self.fanouts}, tile={self.tile_rows}x{self.tile_cols}, "
                f"warps_per_block={self.warps_per_block}, "
                f"cache_size={len(self._bcsr_cache)})")

    # 输出平均统计
    def print_avg_timing(self):
        class AvgTimingObj: pass

        print("\n" + "-"*60)
        print(" [Average Sampling Performance Summary (All Epochs)]")
        print("-"*60)

        for layer_id in sorted(self.time_tracker.keys()):
            stats = self.time_tracker[layer_id]
            cnt = stats['count']
            if cnt == 0: continue

            avg_t = AvgTimingObj()
            avg_t.phase1_sampling_time       = stats['p1_sample'] / cnt
            avg_t.phase2_mapping_time        = stats['p2_map'] / cnt
            avg_t.phase3_recompression_time  = stats['p3_recomp'] / cnt
            avg_t.total_time                 = stats['total'] / cnt

            self._print_performance_summary(avg_t, layer_id)

    def sample_partitions(self, g: BCSRGraph, partition_ids: torch.Tensor) -> BCSRGraph:
        """BAPS 核心逻辑"""
        if self.partition_book is None:
            raise ValueError("Partition book is missing! Please set sampler.partition_book.")

        device = partition_ids.device
        TILE_ROWS = self.tile_rows
        TILE_COLS = self.tile_cols
        TILE_AREA = TILE_ROWS * TILE_COLS
        
        kernel_idx_dtype = torch.int32
        kernel_offset_dtype = torch.int64
        
        pids = partition_ids.cpu().numpy()
        selected_parts = [self.partition_book[pid] for pid in pids]

        node_map = torch.full((g.num_nodes,), -1, dtype=torch.int64, device=device)
        
        current_local_node = 0
        sliced_cols_list = []
        sliced_lengths_list = []
        sliced_vals_list = []
        has_values = (g.values_condensed is not None)

        for p_info in selected_parts:
            start_node = p_info['start_idx']
            end_node = p_info['end_idx'] 
            node_count = end_node - start_node
            
            local_ids = torch.arange(current_local_node, current_local_node + node_count, 
                                     dtype=torch.int64, device=device)
            node_map[start_node : end_node] = local_ids
            current_local_node += node_count

            start_win = start_node // TILE_ROWS
            end_win = end_node // TILE_ROWS
            
            chunk_offsets = g.window_offset[start_win : end_win + 1]
            chunk_lengths = chunk_offsets[1:] - chunk_offsets[:-1]
            
            tile_start_idx = chunk_offsets[0].item()
            tile_end_idx = chunk_offsets[-1].item()
            
            chunk_cols = g.original_col_indices[tile_start_idx * TILE_COLS : tile_end_idx * TILE_COLS]
            
            sliced_cols_list.append(chunk_cols)
            sliced_lengths_list.append(chunk_lengths)
            
            if has_values:
                chunk_vals = g.values_condensed[tile_start_idx * TILE_AREA : tile_end_idx * TILE_AREA]
                sliced_vals_list.append(chunk_vals)

        final_cols_temp = torch.cat(sliced_cols_list) 
        all_lengths = torch.cat(sliced_lengths_list)
        all_vals = torch.cat(sliced_vals_list) if has_values else None

        if final_cols_temp.device != device: final_cols_temp = final_cols_temp.to(device)
        if all_lengths.device != device: all_lengths = all_lengths.to(device)
        if has_values and all_vals.device != device: all_vals = all_vals.to(device)
        
        final_cols_working = final_cols_temp.clone()
        mask_existing = (final_cols_working != -1)
        valid_global_ids = final_cols_working[mask_existing].long()
        mapped_local_ids = node_map[valid_global_ids]
        final_cols_working[mask_existing] = mapped_local_ids.to(final_cols_working.dtype)

        # [新增修复] 处理 Padding (-1)
        # 1. 识别无效位置
        mask_padding = ~mask_existing
        
        # 2. 将无效索引设为 0 (安全占位，防止 CUDA 越界访问 feat[-1])
        final_cols_working[mask_padding] = 0 
        
        # 3. 将无效位置对应的权重设为 0.0 (防止计算污染)
        # 确保即使索引冲突，其对结果的贡献也为 0
        if has_values:
             # values 维度是 [num_tiles * tile_rows * tile_cols]
             # mask   维度是 [num_tiles * tile_cols]
             # 需要利用 view 和 expand 进行维度对齐
             num_tiles = final_cols_working.size(0) // TILE_COLS
             
             vals_view = all_vals.view(num_tiles, TILE_ROWS, TILE_COLS)
             mask_pad_view = mask_padding.view(num_tiles, TILE_COLS)
             
             # 扩展 mask: [Tiles, Cols] -> [Tiles, Rows, Cols]
             mask_expanded = mask_pad_view.unsqueeze(1).expand(-1, TILE_ROWS, -1)
             
             # 原地修改权重为 0
             vals_view.masked_fill_(mask_expanded, 0.0)
        
        final_cols = final_cols_working.to(kernel_idx_dtype)
        
        new_offsets = torch.zeros(len(all_lengths) + 1, dtype=kernel_offset_dtype, device=device)
        new_offsets[1:] = torch.cumsum(all_lengths, dim=0).to(kernel_offset_dtype)
        
        subgraph = BCSRGraph.from_components(
            window_offset=new_offsets,      
            original_col_indices=final_cols,
            values_condensed=all_vals,
            num_rows=current_local_node,
            num_cols=current_local_node, 
            tile_rows=TILE_ROWS,
            tile_cols=TILE_COLS,
            device=device
        )
        
        num_windows = len(all_lengths)
        subgraph.active_windows = torch.arange(num_windows, device=device, dtype=torch.int32)
        subgraph.is_local_input = True
        
        return subgraph


    def repack_subgraph(self, g: BCSRGraph, window_ids: torch.Tensor, max_external_ratio: float = 5.0) -> BCSRGraph:
        """
        [方案 B 增强版] Halo-Aware Subgraph Repacking with OOM Guard.
        
        Args:
            g: 全局图
            window_ids: 当前 Batch 涉及的 Window ID
            max_external_ratio: 外部节点数量限制倍率。
                                限制 External Nodes <= Internal Nodes * Ratio
                                防止 Super Hub 导致显存爆炸。
        """
        device = window_ids.device
        window_ids, _ = torch.sort(window_ids)
        num_windows = window_ids.size(0)
        num_internal_nodes = num_windows * self.tile_rows

        # 1. 准备数据访问指针
        starts = g.window_offset[window_ids]
        ends = g.window_offset[window_ids + 1]
        lengths = ends - starts 
        
        new_window_offset = torch.zeros(num_windows + 1, dtype=torch.int64, device=device)
        new_window_offset[1:] = torch.cumsum(lengths, dim=0)
        
        # 2. 批量提取所有原始列索引 (Gather)
        # 这一步是把所有邻居（无论内部外部）都抓出来
        expanded_win_indices = torch.repeat_interleave(torch.arange(num_windows, device=device), lengths)
        global_tile_seq = torch.arange(new_window_offset[-1].item(), device=device)
        # 计算 gather 索引
        new_win_starts_expanded = new_window_offset[expanded_win_indices]
        intra_window_offsets = global_tile_seq - new_win_starts_expanded
        old_win_starts_expanded = starts[expanded_win_indices]
        gather_tile_indices = old_win_starts_expanded + intra_window_offsets
        
        # 原始列索引 (Global Node IDs, 包含 -1)
        orig_cols_view = g.original_col_indices.view(-1, self.tile_cols)
        raw_col_indices = orig_cols_view[gather_tile_indices].view(-1)
        
        # 原始权重 (如果有)
        raw_values = None
        if g.values_condensed is not None:
            orig_vals_view = g.values_condensed.view(-1, self.tile_rows, self.tile_cols)
            raw_values = orig_vals_view[gather_tile_indices].view(-1)

        # 3. 识别 Internal Nodes (必须保留)
        # 根据 window_ids 生成内部节点的全集
        # expand: [win0, win1] -> [win0_r0...win0_r15, win1_r0...]
        expanded_win = torch.repeat_interleave(window_ids, self.tile_rows)
        offsets = torch.tile(torch.arange(self.tile_rows, device=device), (num_windows,))
        internal_nodes_global = expanded_win * self.tile_rows + offsets
        
        # 4. 识别 External Nodes (Halo)
        # 过滤掉 -1
        mask_valid_edge = (raw_col_indices != -1)
        valid_neighbors = raw_col_indices[mask_valid_edge].long()
        
        # 找出所有涉及的 Unique Neighbors
        unique_neighbors = torch.unique(valid_neighbors)
        
        # 分离出 External (Unique Neighbors - Internal Nodes)
        # 这是一个集合差集操作。
        # 既然 Internal Nodes 是有序且分块连续的，我们可以用 mask 标记
        # 但为了通用性，我们使用 boolean mask
        
        # 技巧：构建一个全图 mask (如果显存允许) 或者使用 searchsorted
        # 考虑到 Reddit 只有 20万节点，全图 Mask 很便宜 (200KB)
        is_internal_mask = torch.zeros(g.num_nodes, dtype=torch.bool, device=device)
        is_internal_mask[internal_nodes_global] = True
        
        # 标记哪些 neighbor 是 external
        # 注意：这里可能 valid_neighbors 里有越界值(如果预处理没做好)，加个 clamp 保护
        valid_neighbors = torch.clamp(valid_neighbors, max=g.num_nodes-1)
        
        # 只有那些不是 internal 的 unique neighbor 才是 external candidates
        is_external_node = ~is_internal_mask[unique_neighbors]
        external_candidates = unique_neighbors[is_external_node]
        
        # 5. [OOM Guard] 安全熔断机制
        num_budget = int(num_internal_nodes * max_external_ratio)
        if external_candidates.size(0) > num_budget:
            # 随机采样保留一部分
            perm = torch.randperm(external_candidates.size(0), device=device)
            kept_external = external_candidates[perm[:num_budget]]
            # 重新排序以便后续 searchsorted
            kept_external, _ = torch.sort(kept_external)
        else:
            kept_external = external_candidates
            
        # 6. 构建最终的 Input Feature Nodes (Features Dictionary)
        # 顺序：[Internal Nodes (顺序必须固定), External Nodes (顺序随意)]
        # Internal Nodes 必须排在前面，且顺序对应 Output Window，
        # 这样模型输出的前 num_internal 行就直接对应 Loss 需要的行。
        final_input_nodes = torch.cat([internal_nodes_global, kept_external])
        
        # 7. 重构列索引 (Reindexing)
        # 我们需要把 raw_col_indices (Global) 映射到 final_input_nodes 的下标 (Local)
        
        # 构建映射表: Global ID -> Local ID
        # 为了速度，我们使用稀疏映射或全图映射
        # Reddit 节点数较小，用全图数组映射最快
        global_to_local = torch.full((g.num_nodes,), -1, dtype=torch.int32, device=device)
        global_to_local[final_input_nodes] = torch.arange(final_input_nodes.size(0), dtype=torch.int32, device=device)
        
        # 查表
        # 无效边(-1) 映射结果是 global_to_local[-1] -> 最后一个元素的映射值，或者是 -1 (如果初始化正确)
        # 为了安全，我们手动处理 -1
        new_col_indices_flat = torch.full_like(raw_col_indices, -1)
        
        # 只对有效边进行重映射
        valid_indices_mask = (raw_col_indices != -1)
        valid_global_ids = raw_col_indices[valid_indices_mask].long()
        
        mapped_ids = global_to_local[valid_global_ids]
        
        # 8. 处理被 OOM Guard 丢弃的边
        # 如果某个外部节点被丢弃了，mapped_ids 会是 -1
        # 我们需要保留 mapped_ids != -1 的边
        mask_kept_edges = (mapped_ids != -1)
        
        # 写入结果
        # 注意：valid_indices_mask 里的第 k 个 True 位置，对应 mask_kept_edges 的第 k 个值
        # 这需要两步索引
        final_write_pos = torch.nonzero(valid_indices_mask, as_tuple=True)[0][mask_kept_edges]
        final_write_val = mapped_ids[mask_kept_edges]
        
        new_col_indices_flat[final_write_pos] = final_write_val
        
        # Reshape
        new_col_indices = new_col_indices_flat.view(-1, self.tile_cols)
        
        # 9. 处理 Values (清洗被丢弃的边权重)
        if raw_values is not None and isinstance(raw_values, torch.Tensor):
             # [修复] 解决维度不匹配问题
             # mask_garbage 对应列索引 [Total_Tiles * Tile_Cols]
             # raw_values 对应具体值 [Total_Tiles * Tile_Rows * Tile_Cols]
             
             # 1. 找出无效的列 (Padding 或被 Budget 丢弃的 Halo)
             mask_garbage_cols = (new_col_indices_flat == -1)
             
             # 2. 计算 Tile 数量
             total_tiles = new_col_indices_flat.size(0) // self.tile_cols
             
             # 3. 变形 Mask: [Total_Tiles, Tile_Cols]
             mask_view = mask_garbage_cols.view(total_tiles, self.tile_cols)
             
             # 4. 扩展 Mask 到所有行: [Total_Tiles, Tile_Cols] -> [Total_Tiles, Tile_Rows, Tile_Cols]
             # 含义：如果某列是无效的，那么该列在 Tile 内的所有 16 行数值都应设为 0
             mask_expanded = mask_view.unsqueeze(1).expand(-1, self.tile_rows, -1)
             
             # 5. 变形 Values 以匹配 Mask
             vals_view = raw_values.view(total_tiles, self.tile_rows, self.tile_cols)
             
             # 6. 原地清洗 (In-place Zeroing)
             vals_view.masked_fill_(mask_expanded, 0.0)
        
        # 10. 组装子图 (Halo Graph)
        subgraph = BCSRGraph.from_components(
            window_offset=new_window_offset,
            original_col_indices=new_col_indices,
            values_condensed=raw_values,
            num_rows=num_internal_nodes,      
            num_cols=final_input_nodes.size(0), # N_in
            tile_rows=self.tile_rows,
            tile_cols=self.tile_cols,
            device=device
        )
        
        subgraph.original_node_ids = final_input_nodes 
        subgraph.original_window_ids = window_ids
        subgraph.is_local_input = True
        subgraph.num_internal_nodes = num_internal_nodes 

        # [Fix] 强力清洗 Halo Graph 的 NaN
        if raw_values is not None:
             # 1. 再次执行 -1 位置清零
             mask_garbage_cols = (new_col_indices == -1).view(-1)
             
             # [修复] 获取正确的 Tile 数量
             total_tiles = new_col_indices.size(0)
             
             mask_view = mask_garbage_cols.view(total_tiles, self.tile_cols)
             mask_expanded = mask_view.unsqueeze(1).expand(-1, self.tile_rows, -1)
             
             vals_view = raw_values.view(total_tiles, self.tile_rows, self.tile_cols)
             vals_view.masked_fill_(mask_expanded, 0.0)
             
             # 2. 检查并替换 NaN/Inf
             if torch.isnan(vals_view).any():
                 vals_view.nan_to_num_(0.0)

        # =========================================================
        # [新增] 生成 Inner Graph (用于 Layer 1+)
        # 逻辑：克隆 Halo Graph，但将所有指向 Halo Node (index >= num_internal) 的边屏蔽掉
        # =========================================================
        
        # 1. 克隆列索引
        inner_col_indices = new_col_indices.clone()
        
        # 2. 识别 Halo Edges (Local Index >= num_internal_nodes 的都是外部节点)
        # 注意：new_col_indices 中可能还有 -1
        mask_halo = (inner_col_indices >= num_internal_nodes)
        
        # 3. 将 Halo Edges 设为 -1 (屏蔽)
        inner_col_indices[mask_halo] = -1
        
        # 4. 处理 Values (如果有)
        inner_values = None
        if raw_values is not None:
            inner_values = raw_values.clone()
            
            # 同样需要清洗被屏蔽的边
            # Mask 对应的是 mask_halo (shape: [tiles, cols])
            
            # [修复] inner_col_indices 已经是 [Num_Tiles, Cols] 形状
            total_tiles = inner_col_indices.size(0) 
            
            # mask_halo 本身已经是 [Num_Tiles, Cols]，可以直接用，或者 view 确保一下
            mask_halo_view = mask_halo.view(total_tiles, self.tile_cols)
            mask_halo_expanded = mask_halo_view.unsqueeze(1).expand(-1, self.tile_rows, -1)
            
            inner_vals_view = inner_values.view(total_tiles, self.tile_rows, self.tile_cols)
            inner_vals_view.masked_fill_(mask_halo_expanded, 0.0)
            
        # 5. 构建 Inner Graph
        # 关键：num_cols 必须限制为 num_internal_nodes
        inner_graph = BCSRGraph.from_components(
            window_offset=new_window_offset, # 共享 Offset
            original_col_indices=inner_col_indices,
            values_condensed=inner_values,
            num_rows=num_internal_nodes,
            num_cols=num_internal_nodes, # N_out (Square Matrix)
            tile_rows=self.tile_rows,
            tile_cols=self.tile_cols,
            device=device
        )
        
        # 挂载 Inner Graph 到主图对象上
        subgraph.inner_graph = inner_graph

        return subgraph


    def sample_dps_online(self, g: BCSRGraph, seed_window_ids: torch.Tensor, fanouts: List[int] = [5, 5]) -> BCSRGraph:
        """[DPS 在线接口] 探测 (Probe) + 重组 (Repack)"""
        device = seed_window_ids.device
        
        # 1. 将 Window ID 转为 Node ID
        expanded_win = torch.repeat_interleave(seed_window_ids, self.tile_rows)
        offsets = torch.tile(torch.arange(self.tile_rows, device=device), (len(seed_window_ids),))
        seed_nodes = expanded_win * self.tile_rows + offsets
        
        # [Fix 2] 过滤掉 Padding 区域产生的非法节点 ID (即 >= g.num_nodes 的部分)
        # 否则传入 sample_layers 会导致 CUDA Kernel 越界访问
        if seed_nodes.numel() > 0:
            valid_mask = seed_nodes < g.num_nodes
            seed_nodes = seed_nodes[valid_mask]
        
        if seed_nodes.numel() == 0:
            # 返回空图避免报错
            return self.repack_subgraph(g, torch.tensor([], device=device, dtype=torch.long))
        
        # 2. 探测 (Probe)
        probe_sampler = BCSRSampler(
            fanouts=fanouts, 
            tile_rows=self.tile_rows, 
            tile_cols=self.tile_cols,
            warps_per_block=self.warps_per_block,
            verbose=False 
        )
        
        # 执行采样
        layers_data = probe_sampler.sample_layers(g, seed_nodes)
        
        # 3. 提取 Active Windows
        all_active_windows = []
        for layer_graph in layers_data:
            if hasattr(layer_graph, 'active_windows'):
                all_active_windows.append(layer_graph.active_windows)
        
        if not all_active_windows:
            unique_windows = seed_window_ids
        else:
            combined = torch.cat(all_active_windows)
            unique_windows = torch.unique(combined)

        valid_win_mask = (unique_windows >= 0) & (unique_windows < g.num_windows)
        unique_windows = unique_windows[valid_win_mask]
        
        if unique_windows.numel() == 0:
            unique_windows = seed_window_ids 
            
        # 4. 重组 (Repack)
        subgraph = self.repack_subgraph(g, unique_windows)
        
        return subgraph


