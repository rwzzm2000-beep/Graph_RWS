# Graph_RWS/layers/layer.py
"""
基于 BCSR 的自定义图神经网络层
- 使用自定义 Tensor Core 加速的 SpMM (spmm_tc)
- 提供 autograd 支持
- 直接使用 BCSRGraph
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from ..data import BCSRGraph
from ..helper import record_spmm_time

# 导入自定义 CUDA 扩展中的 SpMM 接口
try:
    from ..core import spmm_tc, spmm_transpose, reindex_bcsr
except ImportError:
    print("=" * 60)
    print("WARNING: CUDA kernels not found!")
    print("Please compile the CUDA extension first:")
    print("  cd core && python setup.py build_ext --inplace")
    print("=" * 60)
    spmm_tc = None


class BCSRSPMMFunction(Function):
    """
    使用自定义的 spmm_tc CUDA kernel 实现 SpMM 操作，并支持自动求导：
    output = A @ X @ W
    其中 A 为 BCSRGraph 表示的稀疏邻接/权重矩阵。
    
    Forward 使用 A 直接计算。
    Backward 使用 spmm_transpose (基于 A 隐式计算 A^T @ G)。
    """
    @staticmethod
    def forward(ctx, graph, features, weight, warps_per_block):
        # X @ W
        x_mult_w = torch.matmul(features, weight)
        
        # H_out = A @ (X @ W)
        output, timing = spmm_tc(
            graph.window_offset,
            graph.original_col_indices,
            graph.values_condensed,
            x_mult_w,
            warps_per_block
        )

        record_spmm_time(timing)
        
        # 保存上下文供 backward 使用
        ctx.save_for_backward(features, weight)
        ctx.graph = graph 
        ctx.warps_per_block = warps_per_block

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, weight = ctx.saved_tensors
        graph = ctx.graph

        # 1. 计算聚合梯度: Grad_Aggr = A^T @ Grad_Output
        # 我们使用 spmm_transpose 算子，它会读取 graph (A) 并通过 atomicAdd 实现转置乘法
        # 需要传入 features.shape[0] 作为 num_input_nodes (即 A^T 的行数，或 A 的列数)
        # 这对于 Local Input 模式 (Partial Graph) 尤为重要
        num_input_nodes = features.shape[0]

        grad_input_aggr = spmm_transpose(
            graph.window_offset,
            graph.original_col_indices,
            graph.values_condensed,
            grad_output.contiguous(), # 确保内存连续
            num_input_nodes
        )
        
        # spmm_transpose 返回的 grad_input_aggr 形状即为 [N_in, D_out]
      
        grad_features = None
        grad_weight = None

        # 2. dX = (A^T @ dY) @ W^T
        if ctx.needs_input_grad[1]:
            grad_features = torch.matmul(grad_input_aggr, weight.t())

        # 3. dW = X^T @ (A^T @ dY)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.matmul(features.t(), grad_input_aggr)

        return None, grad_features, grad_weight, None


class BCSRGraphConv(nn.Module):
    """
    GraphSAGE-style Convolution (aggregation only).
    Nonlinearities/normalization/dropout are handled at the model level.
    """
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.5, warps_per_block: int = 8):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.warps_per_block = warps_per_block

        # GraphSAGE 两个权重矩阵
        # 为了兼容性和显存效率，我们采用 Sum(Linear(Neighbor), Linear(Self)) 的方式
        # 这在数学上等价于 Linear(Concat(Neighbor, Self))
        self.weight_neigh = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_self = nn.Parameter(torch.Tensor(in_feats, out_feats))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_neigh)
        nn.init.xavier_uniform_(self.weight_self)

    def forward(self, graph, features: torch.Tensor, features_target: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播函数 - GraphSAGE 架构 (Neighbor + Self)
        
        Args:
            graph: BCSRGraph 对象
            features: 输入特征矩阵 [N_in, D_in]。
                      在全图模式下 N_in = N_nodes。
                      在采样模式下 N_in = Target_Nodes + Neighbor_Nodes。
            features_target: [可选] 仅包含目标节点的特征矩阵 [N_out, D_in]。
                             如果为 None，函数将尝试自动从 features 中切片（假设 Target-Top 布局）。
        
        Returns:
            out: 输出特征矩阵 [N_out, D_out]
        """
        # 0. 检查算子是否可用
        if spmm_tc is None:
            raise ImportError("spmm_tc CUDA kernel is not available. Please compile the core library.")
        
        # -------------------------------------------------------
        # Path A: 邻居特征聚合 (Neighbor Aggregation)
        # -------------------------------------------------------
        # 这里通过 Autograd Function 调用你的自定义算子 spmm_tc
        # 计算公式: H_neigh = A * (X * W_neigh)
        # 输出形状: [N_out, out_feats] (N_out 通常等于 graph.num_rows)
        h_neigh = BCSRSPMMFunction.apply(graph, features, self.weight_neigh, self.warps_per_block)
        
        # -------------------------------------------------------
        # Path B: 自身特征变换 (Self Transformation)
        # -------------------------------------------------------
        # 我们需要确保自身特征 h_self 的行数与 h_neigh (N_out) 一致。
        
        # 获取目标行数 (N_out)
        target_rows = h_neigh.shape[0]
        
        # 准备对齐的输入特征 h_in_self
        if features_target is not None:
            # Case 1: 显式传入了对齐的特征 (推荐，由 model.py 控制)
            h_in_self = features_target
        else:
            # Case 2: 自动对齐 (Fallback)
            # 假设采样器将目标节点放在 features 的最前面 (Target-Top Layout)
            if features.shape[0] >= target_rows:
                h_in_self = features[:target_rows]
            else:
                # 异常保护：如果 features 行数比要求的输出还少 (可能是 Padding 或 Corner case)
                # 进行零填充以匹配维度
                pad_len = target_rows - features.shape[0]
                h_in_self = F.pad(features, (0, 0, 0, pad_len))

        # 计算 X_target * W_self
        # 输出形状: [N_out, out_feats]
        h_self = torch.matmul(h_in_self, self.weight_self)

        # 维度对齐修复
        # 问题：h_neigh 可能是 2720 (物理对齐), h_self 是 2708 (逻辑真实)
        # 解决：将 h_neigh 裁剪到和 h_self 一样长
        if h_neigh.shape[0] > h_self.shape[0]:
            h_neigh = h_neigh[:h_self.shape[0]]
        elif h_neigh.shape[0] < h_self.shape[0]:
            # 防御性编程：万一 h_neigh 比 h_self 还短 (极少见)，则填充
            pad_len = h_self.shape[0] - h_neigh.shape[0]
            h_neigh = F.pad(h_neigh, (0, 0, 0, pad_len))

        # # 3. 维度对齐 (防御性裁剪)
        # min_len = min(h_neigh.shape[0], h_self.shape[0])
        # h_neigh = h_neigh[:min_len]
        # h_self = h_self[:min_len]
        
        # -------------------------------------------------------
        # Merge: 合并两路特征
        # -------------------------------------------------------
        # Aggregation (Sum 等价于 Concat 后 Linear)
        return h_neigh + h_self
