# Graph_RWS/models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Union
import copy

from ..data import BCSRGraph
from ..layers import BCSRGraphConv
from ..core import reindex_bcsr

class GCN_BCSR(nn.Module):
    """
    基于BCSR格式的图卷积网络（GCN） - 优化版
    集成：双向重索引、梯度检查点、UVA 模式适配
    """
    
    def __init__(self,
                 in_feats: int,
                 hidden_size: int,
                 num_classes: int,
                 num_layers: int,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 residual: bool = False,
                 warps_per_block: int = 8,
                 use_checkpoint: bool = True):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"
        
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.warps_per_block = warps_per_block
        self.use_checkpoint = use_checkpoint
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.layers = nn.ModuleList()
        # Input Layer
        self.layers.append(BCSRGraphConv(in_feats, hidden_size, warps_per_block=warps_per_block))
        # Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(BCSRGraphConv(hidden_size, hidden_size, warps_per_block=warps_per_block))
        # Output Layer
        self.layers.append(BCSRGraphConv(hidden_size, num_classes, warps_per_block=warps_per_block))
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.norms.append(nn.LayerNorm(hidden_size))
        
        if residual:
            self.residual_projs = nn.ModuleList()
            for i in range(num_layers - 1):
                in_dim = in_feats if i == 0 else hidden_size
                out_dim = hidden_size
                self.residual_projs.append(nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None)
        else:
            self.residual_projs = None
        
        self._init_parameters()
    
    def _init_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if self.residual_projs is not None:
            for proj in self.residual_projs:
                if proj is not None:
                    nn.init.kaiming_uniform_(proj.weight, nonlinearity='relu')
        for norm in self.norms:
            norm.reset_parameters()
    
    def _ensure_graph_list(self, graphs: Union[BCSRGraph, List[BCSRGraph]]) -> List[BCSRGraph]:
        if isinstance(graphs, BCSRGraph):
            # [关键修复] 使用 shallow copy 创建独立的图对象封装
            # 这样我们在 forward 中修改 g.original_col_indices 时，不会影响其他层
            return [copy.copy(graphs) for _ in range(len(self.layers))]
        if isinstance(graphs, list):
            assert len(graphs) == len(self.layers)
            return graphs
        raise TypeError(f"graphs must be BCSRGraph or List[BCSRGraph], got {type(graphs)}")

    def _sanitize_and_reindex(self, layer_idx, curr_g, prev_g, h_in_shape):
        """
        [优化版] 使用 CUDA Kernel (reindex_bcsr) 进行双向重索引
        """
        # [新增] 全局跳过检查
        # 如果图被标记为 local_input，说明是采样好的子图，直接跳过所有 Reindex 逻辑
        if hasattr(curr_g, 'is_local_input') and curr_g.is_local_input:
            curr_g.num_cols = h_in_shape[0] # 仅更新元数据
            return
        
        # === A. Forward Graph 处理 ===
        fwd_g = curr_g.fwd if hasattr(curr_g, 'fwd') else curr_g
        
        # Layer 0: Global -> Local (通常输入是全图特征)
        if layer_idx == 0:
            # [新增] 兼容性检查：如果 Pipeline 已经把图转成了 Local Input 模式，则跳过处理
            if hasattr(curr_g, 'is_local_input') and curr_g.is_local_input:
                # 已经是 Local Indices (0..N)，直接通过
                fwd_g.num_cols = h_in_shape[0]
                pass
            else:
                # 原始逻辑：保持 Global Index，清洗越界边
                if hasattr(fwd_g, 'original_col_indices'):
                    bad_mask = (fwd_g.original_col_indices == -1) | (fwd_g.original_col_indices >= h_in_shape[0])
                    fwd_g.original_col_indices.masked_fill_(bad_mask, 0)
                    if hasattr(fwd_g, 'values_condensed'):
                        mask_3d = bad_mask.view(-1, fwd_g.tile_cols).unsqueeze(1)
                        mask_expanded = mask_3d.expand(-1, fwd_g.tile_rows, -1)
                        mask_flat = mask_expanded.reshape(-1)
                        fwd_g.values_condensed.view(-1).masked_fill_(mask_flat, 0.0)
        else:
            # Layer > 0: Local -> Local
            # 使用 prev_g.active_windows (即上一层的输出/当前层的输入) 作为重索引基准
            if hasattr(prev_g, 'active_windows') and prev_g.active_windows is not None:
                # 捕获返回值 new_cols
                new_cols = reindex_bcsr(
                    fwd_g.original_col_indices, # In
                    fwd_g.values_condensed,     # In/Out (Values 原地剪枝)
                    prev_g.active_windows,      # Search Target (上一层的 Active Windows)
                    fwd_g.tile_rows,
                    fwd_g.tile_cols
                )
                # 将 Local Index 更新回图对象
                # 这样 layer.py 就不用再做 Reindex 了
                fwd_g.original_col_indices = new_cols

            # 更新列数元数据
            fwd_g.num_cols = h_in_shape[0]

    def _get_residual(self, layer_idx, curr_g, prev_g, h_in, target_rows):
        """
        [残差提取] 适配 UVA 模式
        """
        if h_in.shape[0] == target_rows:
            return h_in
            
        device = h_in.device
        
        # Layer 0
        if layer_idx == 0:
            # [新增] UVA 模式下的安全措施
            # 如果当前图是 Local Input (特征已被裁剪)，且有 Active Windows (全局ID)
            # 此时无法简单地做 Global->Local 映射，因为 h_in 已经丢失了全局映射关系
            # 策略：直接跳过 Layer 0 的残差连接（返回 0），这在大图训练中是安全的权衡
            if hasattr(curr_g, 'is_local_input') and curr_g.is_local_input:
                return torch.zeros((target_rows, h_in.shape[1]), device=device)

            if not hasattr(curr_g, 'active_windows') or curr_g.active_windows is None:
                if target_rows > h_in.shape[0]:
                    return F.pad(h_in, (0, 0, 0, target_rows - h_in.shape[0]))
                return h_in[:target_rows]
            else:
                # Global -> Local Gather
                t_rows = curr_g.tile_rows
                gather_idx = (curr_g.active_windows.unsqueeze(1) * t_rows + 
                              torch.arange(t_rows, device=device).unsqueeze(0)).view(-1)
                
                num_src = h_in.shape[0]
                valid = gather_idx < num_src
                safe_idx = torch.where(valid, gather_idx, torch.tensor(0, device=device))
                res = h_in[safe_idx]
                mask_view = (~valid).unsqueeze(1)
                res.masked_fill_(mask_view, 0.0)
                return res

        # Layer > 0
        else:
            if hasattr(curr_g, 'active_windows') and curr_g.active_windows is not None:
                t_rows = curr_g.tile_rows
                curr_wins = curr_g.active_windows
                prev_wins = prev_g.active_windows
                
                win_indices = torch.searchsorted(prev_wins, curr_wins)
                max_idx = max(0, prev_wins.size(0) - 1)
                win_indices = torch.clamp(win_indices, min=0, max=max_idx)
                
                gather_idx = (win_indices.unsqueeze(1) * t_rows + 
                              torch.arange(t_rows, device=device).unsqueeze(0)).view(-1)
                return h_in[gather_idx]
            
            # [修改开始] ------------------------------------------------
            # 处理没有 active_windows 的情况 (如 DGL 实验模式)
            
            # Case 1: 需要 Padding
            if target_rows > h_in.shape[0]:
                return F.pad(h_in, (0, 0, 0, target_rows - h_in.shape[0]))
            
            # Case 2: 需要 Slicing (关键修复!)
            # 如果是 Local Input 模式 (DGL Block)，Target 节点总是在最前面
            # 所以直接截取前 target_rows 行
            if target_rows < h_in.shape[0]:
                return h_in[:target_rows]
                
            return h_in
            # [修改结束] ------------------------------------------------

    def forward(self, graphs: Union[BCSRGraph, List[BCSRGraph]], features: torch.Tensor) -> torch.Tensor:
        if features.device.type != 'cuda':
            features = features.to('cuda', non_blocking=True)
        
        graph_list = self._ensure_graph_list(graphs)
        h = features
        
        for l, (layer, g) in enumerate(zip(self.layers, graph_list)):
            # 1. Reindex (必须在 checkpoint 之外，因为涉及图结构修改)
            prev_g = graph_list[l-1] if l > 0 else None
            self._sanitize_and_reindex(l, g, prev_g, h.shape)
            
            # =======================================================
            # 准备对齐的 Self 特征
            # 这里的逻辑与 _get_residual 完全一致：从 h (N_in) 中提取出 h_target (N_out)
            target_rows = g.num_rows # 或者 g.num_windows * g.tile_rows
            
            # 复用 _get_residual 函数来做 Gather/Slice 操作
            # 注意：_get_residual 本意是为残差连接服务的，但它做的"对齐"工作正是我们需要给 h_self 做的
            h_target = self._get_residual(l, g, prev_g, h, target_rows)

            # 2. 定义层计算函数
            # 将 h_target 传入 layer
            def run_curr_layer(h_curr, layer=layer, g=g, h_tgt=h_target):
                return layer(g, h_curr, features_target=h_tgt)

            # 3. 执行卷积 (使用 Checkpoint 或 直接执行)
            # Checkpoint 条件：开启且是训练模式且 tensor 需要梯度
            if self.use_checkpoint and self.training and h.requires_grad:
                h_out = checkpoint(run_curr_layer, h, use_reentrant=False)
            else:
                h_out = run_curr_layer(h)
            
            # 4. 残差
            if self.residual and l < len(self.layers) - 1:
                h_res = h
                if self.residual_projs is not None:
                    proj = self.residual_projs[l]
                    if proj is not None:
                        h_res = proj(h_res)
                
                h_res_aligned = self._get_residual(l, g, prev_g, h_res, h_out.shape[0])
                h_out = h_out + h_res_aligned
            
            h = h_out
            
            # 5. 激活/Dropout
            if l < len(self.layers) - 1:
                h = self.activation(h)
                if self.norms:
                    h = self.norms[l](h)
                if self.dropout_layer is not None:
                    h = self.dropout_layer(h)
                    
        return h

    def inference(self, graph: BCSRGraph, features: torch.Tensor, batch_size: int = 0) -> torch.Tensor:
        """推理模式：通常是全图，不需要 Reindex"""
        if features.device.type != 'cuda':
            features = features.to('cuda', non_blocking=True)
        h = features
        
        # 为了安全，Layer 0 还是做一下 sanitize (清洗越界边)
        # 伪造一个空的 prev_g
        g_copy = copy.copy(graph)
        self._sanitize_and_reindex(0, g_copy, None, h.shape)
        
        for l, layer in enumerate(self.layers):
            h = layer(g_copy, h)
            if l < len(self.layers) - 1:
                h = self.activation(h)
                if self.norms:
                    h = self.norms[l](h)
        return h
