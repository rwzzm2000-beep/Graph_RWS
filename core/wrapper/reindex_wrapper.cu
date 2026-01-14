// PyTorch library
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <vector>

#include "../include/reindex.cuh" 
#include "../helper/macros.h"


using namespace zhao;


/**
 * @brief BCSR 重索引 CUDA 接口
 *
 * @param col_indices 原始全局列索引 (Int32)，将被原地修改用于剪枝
 * @param values_condensed 权重值 (Float32)，将被原地修改用于剪枝
 * @param active_windows 活跃窗口列表 (Int64/LongLong)，必须有序
 * @param tile_rows 每个Tile的行数
 * @param tile_cols 每个Tile的列数
 *
 * @return torch::Tensor 重索引后的局部列索引 (Int32)
 */
torch::Tensor reindex_bcsr_cuda(
    const torch::Tensor& col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& active_windows,
    int tile_rows,
    int tile_cols)
{
    // 1. 准备输出 Tensor (非原地修改)
    // 使用 empty_like 避免不必要的内存拷贝 (Clone)
    torch::Tensor out_col_indices = torch::empty_like(col_indices);

    // 2. 获取数据指针
    const int* d_in_cols = GET_CONST_PTR(int, col_indices);
    float* d_values = GET_PTR(float, values_condensed); // 原地修改
    const int* d_active_windows = GET_CONST_PTR(int, active_windows);
    int* d_out_cols = GET_PTR(int, out_col_indices);    // 输出

    // 3. 获取维度信息
    int num_indices = col_indices.numel();
    int num_active = active_windows.size(0);

    // 4. 配置 Kernel
    const int threads = 256;
    const int blocks = (num_indices + threads - 1) / threads;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 5. 启动 Kernel
    reindex_and_prune_kernel<<<blocks, threads, 0, stream>>>(
        d_in_cols,
        d_values,
        d_active_windows,
        num_indices,
        num_active,
        tile_rows,
        tile_cols,
        d_out_cols // 输出放在最后
    );

    // 6. 返回结果
    return out_col_indices;
}

