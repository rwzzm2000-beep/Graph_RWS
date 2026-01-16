# Graph_RWS/data/bcsr.py
"""
BCSR (Blocked Compressed Sparse Row) 图数据结构
- 统一命名：仅保留 values_condensed / original_col_indices / window_offset
- 提供必要的构造与转换：from_dgl / from_mtx / from_components
- 内部使用 COO->CSR->BCSR 的转换（_coo_to_csr / _from_csr）
- 显式支持矩形图（num_rows / num_cols）
"""

from typing import Optional, Dict, Any

import torch
import numpy as np
from tqdm import tqdm

try:
    # 尝试从 core 包导入
    from ..core import csr_to_bcsr_cpu
except ImportError:
    raise ImportError("csr_to_bcsr_cpu function is not found. Please check if the core extension is properly compiled.")


class BCSRGraph:
    """
    BCSR格式的图数据结构（GPU上存储的稀疏图表示）

    数据布局：
    - 图按行划分为 windows（每个 window 包含 tile_rows 行）
    - 每个 window 内的边按列划分为 tiles（每个 tile 包含 tile_cols 列）
    - window_offset: [num_windows + 1]，每个 window 的起始 tile 索引（前缀和）
    - original_col_indices: [num_tiles * tile_cols]，每个 tile 的列全局ID（无效填充为 -1）
    - values_condensed: [num_tiles * tile_rows * tile_cols]，tile 展平后的值
    """

    def __init__(self,
                 num_nodes: int,
                 num_edges: int,
                 tile_rows: int,
                 tile_cols: int,
                 window_offset: torch.Tensor,
                 original_col_indices: torch.Tensor,
                 values_condensed: torch.Tensor,
                 device: Optional[torch.device] = None,
                 num_rows: Optional[int] = None,
                 num_cols: Optional[int] = None) -> None:
        # 归一化设备与张量
        self.device = device
        self.window_offset = window_offset.to(device, non_blocking=True).to(torch.int64)
        self.original_col_indices = original_col_indices.to(device, non_blocking=True).to(torch.int32)
        self.values_condensed = values_condensed.to(device, non_blocking=True).to(torch.float32)

        # 基本属性（显式支持矩形图）
        self.num_rows = int(num_rows) if num_rows is not None else int(num_nodes)
        self.num_cols = int(num_cols) if num_cols is not None else int(num_nodes)
        self.num_nodes = int(num_nodes)  # 兼容字段（=num_rows）
        self.num_edges = int(num_edges)
        self.tile_rows = int(tile_rows)
        self.tile_cols = int(tile_cols)

        # 衍生属性
        self.num_windows = int(self.window_offset.shape[0] - 1)
        self.num_tiles = None

    # ---------------- 构造/转换接口 ----------------
    @classmethod
    def from_dgl(cls, g: Any, tile_rows: int = 16, tile_cols: int = 8) -> 'BCSRGraph':
        """从 DGLGraph 创建 BCSRGraph（无需在此处导入 dgl，仅依赖 g 的接口）。"""
        device = g.device
        num_nodes = g.num_nodes()
        src, dst = g.edges()
        edge_weights = g.edata.get('weight', None)

        # COO -> CSR
        num_edges, row_ptr, col_idx, values = cls._coo_to_csr(
            num_nodes, src, dst, edge_weights, device
        )
        # CSR -> BCSR（方阵：num_rows=num_cols=num_nodes）
        return cls._from_csr(
            num_rows=num_nodes,
            row_ptr=row_ptr,
            col_idx=col_idx,
            edge_values=values,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            device=device,
            num_cols=num_nodes
        )

    @classmethod
    def from_mtx(cls,
                 filepath: str,
                 index_base: int = 0,
                 undirected: bool = False,
                 tile_rows: int = 16,
                 tile_cols: int = 8,
                 device: Optional[torch.device] = None) -> 'BCSRGraph':
        """从 Matrix Market (.mtx) 文件直接创建 BCSRGraph（COO -> CSR -> BCSR）。"""
        import scipy.io as sio
        import scipy.sparse as sps

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        coo = sio.mmread(filepath).tocoo()
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)
        if index_base == 1:
            rows -= 1
            cols -= 1
        if undirected:
            rows = np.concatenate([rows, cols], axis=0)
            cols = np.concatenate([cols, rows[:coo.row.size]], axis=0)

        num_rows = int(max(rows.max(), cols.max()) + 1)
        num_cols = num_rows
        data = np.ones_like(rows, dtype=np.float32)
        csr = sps.csr_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
        row_ptr = torch.from_numpy(csr.indptr).to(device=device, dtype=torch.int64)
        col_idx = torch.from_numpy(csr.indices).to(device=device, dtype=torch.int32)
        values = torch.from_numpy(csr.data).to(device=device, dtype=torch.float32)

        return cls._from_csr(
            num_rows=num_rows,
            row_ptr=row_ptr,
            col_idx=col_idx,
            edge_values=values,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            device=device,
            num_cols=num_cols
        )

    @classmethod
    def from_components(cls,
                        window_offset: torch.Tensor,
                        original_col_indices: torch.Tensor,
                        values_condensed: torch.Tensor,
                        num_rows: int,
                        num_cols: int,
                        tile_rows: int,
                        tile_cols: int,
                        device: Optional[torch.device] = None) -> 'BCSRGraph':
        """根据组件张量直接构造 BCSRGraph（采样/核输出使用）。"""
        if device is None:
            device = window_offset.device
        # 估算边数：按 tile 值非零计数（若 values_condensed 存实际边权，非零近似边数）
        nnz = int(torch.count_nonzero(values_condensed).item()) if values_condensed.numel() > 0 else 0
        return cls(
            num_nodes=int(num_rows),
            num_edges=nnz,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            window_offset=window_offset.to(device, non_blocking=True).to(torch.int64),
            original_col_indices=original_col_indices.to(device, non_blocking=True).to(torch.int32),
            values_condensed=values_condensed.to(device, non_blocking=True).to(torch.float32),
            device=device,
            num_rows=int(num_rows),
            num_cols=int(num_cols)
        )

    # ---------------- 内部转换实现 ----------------
    @classmethod
    def _coo_to_csr(cls,
                    num_rows: int,
                    src: torch.Tensor,
                    dst: torch.Tensor,
                    edge_weights: Optional[torch.Tensor],
                    device: torch.device):
        """使用纯 PyTorch 从 COO 计算 CSR。"""
        sort_key = src.to(torch.long) * num_rows + dst.to(torch.long)
        perm = torch.argsort(sort_key)
        src = src[perm]
        dst = dst[perm]

        num_edges = int(src.numel())
        if edge_weights is not None:
            values = edge_weights[perm].to(device)
            if values.dtype != torch.float32:
                values = values.to(torch.float32)
        else:
            values = torch.ones(num_edges, dtype=torch.float32, device=device)

        row_counts = torch.bincount(src, minlength=num_rows).to(torch.int64)
        row_ptr = torch.zeros(num_rows + 1, dtype=torch.int64, device=device)
        torch.cumsum(row_counts, dim=0, out=row_ptr[1:])
        col_idx = dst.to(torch.int32)
        return num_edges, row_ptr, col_idx, values

    @classmethod
    def _from_csr(cls,
                  num_rows: int,
                  row_ptr: torch.Tensor,
                  col_idx: torch.Tensor,
                  edge_values: torch.Tensor,
                  tile_rows: int,
                  tile_cols: int,
                  device: torch.device,
                  num_cols: Optional[int] = None) -> 'BCSRGraph':
        """从 CSR 转换为 BCSR（核心实现）。"""
        if num_cols is None:
            num_cols = num_rows

        # 1. 准备数据：确保数据在 CPU 上且类型正确
        r_ptr_cpu = row_ptr.to('cpu', dtype=torch.int64)
        c_idx_cpu = col_idx.to('cpu', dtype=torch.int32)
        vals_cpu = edge_values.to('cpu', dtype=torch.float32)

        # 2. 调用 C++ 加速函数 (OpenMP Parallel)
        out_tuple = csr_to_bcsr_cpu(
            int(num_rows),
            r_ptr_cpu,
            c_idx_cpu,
            vals_cpu,
            int(tile_rows),
            int(tile_cols)
        )
        
        window_offset_cpu, original_col_indices_cpu, values_condensed_cpu = out_tuple

        # 3. 构建对象并将结果移回目标设备 (如 GPU)
        num_edges_est = int(col_idx.numel())
        
        return cls(
            num_nodes=int(num_rows),
            num_edges=num_edges_est,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            window_offset=window_offset_cpu.to(device),
            original_col_indices=original_col_indices_cpu.to(device),
            values_condensed=values_condensed_cpu.to(device),
            device=device,
            num_rows=int(num_rows),
            num_cols=int(num_cols)
        )

        
    # ---------------- 实用方法 ----------------
    def to(self, device: torch.device, non_blocking: bool = False) -> 'BCSRGraph':
        """
        移动到指定设备（保留矩形信息）。
        [Fixed] 增加 non_blocking 参数以支持 CUDA 流水线异步传输。
        """
        if device == self.device:
            return self
        
        return BCSRGraph(
            num_nodes=self.num_rows,
            num_edges=self.num_edges,
            tile_rows=self.tile_rows,
            tile_cols=self.tile_cols,
            # 将 non_blocking 参数传递给底层的 Tensor.to()
            window_offset=self.window_offset.to(device, non_blocking=non_blocking),
            original_col_indices=self.original_col_indices.to(device, non_blocking=non_blocking),
            values_condensed=self.values_condensed.to(device, non_blocking=non_blocking),
            device=device,
            num_rows=self.num_rows,
            num_cols=self.num_cols
        )

    def __repr__(self) -> str:
        return (f"BCSRGraph(rows={self.num_rows}, cols={self.num_cols}, windows={self.num_windows}, "
                f"tiles={self.num_tiles}, tile={self.tile_rows}x{self.tile_cols}, device={self.device})")

    # [新增] 增加一个专门用于转换 Block 的方法
    @classmethod
    def from_dgl_block(cls, block: Any, tile_rows: int = 16, tile_cols: int = 8) -> 'BCSRGraph':
        """
        [新增] 专门用于将 DGL 的 MFG (Message Flow Graph/Block) 转换为 BCSR。
        Block 是二部图: (Src Nodes) -> (Dst Nodes)
        """
        device = block.device
        # Block 的行数是 dst_nodes (输出节点), 列数是 src_nodes (输入节点)
        num_rows = block.num_dst_nodes()
        num_cols = block.num_src_nodes()
        
        # 获取边 (block.edges() 返回的是局部索引)
        src, dst = block.edges()
        # 注意: DGL block 边是 (src -> dst)，对应矩阵是 A[dst, src] (如果按行聚合)
        # 或者 A[src, dst] (如果按列聚合)。
        # GraphSAGE 通常是 dst 聚合 src 的信息。
        # 我们这里假设 BCSR 存储的是转置后的关系或者直接对应关系，
        # 关键是：row_ptr 对应的是 dst (目标节点)，col_idx 对应的是 src (邻居)。
        
        # 使用 CSR 转换逻辑
        # DGL Block 内部通常已经是 CSR 或 COO。我们直接取出来。
        # 这里为了通用，先转 COO 再转 CSR (虽然有点慢，但作为实验足够)
        edge_weights = block.edata.get('weight', None) if 'weight' in block.edata else None
        
        # 调用内部 COO->CSR
        # 注意：这里 row 应该是 dst (因为我们要聚合 neighbors 到 dst)
        # col 应该是 src
        num_edges, row_ptr, col_idx, values = cls._coo_to_csr(
            num_rows, dst, src, edge_weights, device  # 注意 dst 和 src 的顺序，视你的 SpMM 逻辑而定
            # 如果你的 SpMM 是 Out = A @ In，且 A 的行代表 Out 节点，那么 Row=Dst, Col=Src。
        )
        
        return cls._from_csr(
            num_rows=num_rows,
            row_ptr=row_ptr,
            col_idx=col_idx,
            edge_values=values,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            device=device,
            num_cols=num_cols # 显式传入列数
        )
