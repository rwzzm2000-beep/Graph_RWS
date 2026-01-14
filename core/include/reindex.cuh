#pragma once

// CUDA runtime
#include <cuda_runtime.h>
#include "../helper/macros.h"



namespace zhao {


/**
 * @brief 重索引 + 剪枝 Kernel
 * * 逻辑：
 * 1. 遍历 in_col_indices 中的每个全局列索引 global_col。
 * 2. 计算它所属的全局窗口 target_win = global_col / tile_rows。
 * 3. 在 active_windows 中二分查找 target_win。
 * 4. 如果找到 (idx)：local_col = idx * tile_rows + (global_col % tile_rows)。
 * 写入 out_col_indices。
 * 5. 如果没找到：说明该邻居不在上一层的输出中 (Ghost Node)。
 * 将 out_col_indices 设为 0 (防止越界)。
 * 将 values 中对应的整列权重置为 0 (Pruning)。
 *
 * @param in_col_indices 原始列索引
 * @param values 权重值
 * @param active_windows 活跃窗口
 * @param num_indices 列索引数
 * @param num_active 活跃窗口数
 * @param tile_rows 每个Tile的行数
 * @param tile_cols 每个Tile的列数
 * @param out_col_indices 输出列索引
 */
static __global__ void reindex_and_prune_kernel(
    const int* __restrict__ in_col_indices,
    float* __restrict__ values,
    const int* __restrict__ active_windows,
    int num_indices,
    int num_active,
    int tile_rows,
    int tile_cols,
    int* __restrict__ out_col_indices)
{
    // 当前线程负责处理的列索引下标
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    // 1. 读取全局列索引
    int global_col = in_col_indices[idx];

    // 2. 计算目标窗口 ID
    int target_win = global_col / tile_rows;

    // 3. 二分查找 (Binary Search)
    // active_windows 是有序的
    int left = 0;
    int right = num_active - 1;
    int found_idx = -1;

    while (left <= right) {
        int mid = (left + right) / 2;
        int win = active_windows[mid]; // 自动走 L1/L2 Cache
        if (win == target_win) {
            found_idx = mid;
            break;
        } else if (win < target_win) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    // 4. 写入结果 & 剪枝
    if (found_idx != -1) {
        // [Found] 映射到 Local ID
        // Local Window ID = found_idx
        // Offset in Tile = global_col % tile_rows
        int local_col = found_idx * tile_rows + (global_col % tile_rows);
        out_col_indices[idx] = local_col;
    } else {
        // [Not Found] 剪枝
        // 1. 将列索引设为 0 (安全值，防止 SpMM 越界访问 features[0])
        out_col_indices[idx] = 0;

        // 2. 将对应的权重值清零 (Pruning)
        // 注意：values 的布局通常是 [Block, Row, Col]
        // col_indices 的长度是 num_active_blocks * tile_cols
        // idx 对应的是第 (idx / tile_cols) 个 Block 的第 (idx % tile_cols) 列
        
        int block_idx = idx / tile_cols;
        int col_in_tile = idx % tile_cols;
        
        // 该 Block 在 values 中的起始位置
        // values 总大小 = num_blocks * tile_rows * tile_cols
        long long val_base = (long long)block_idx * tile_rows * tile_cols;

        // 我们需要把这一列在 Tile 内的所有行 (0 to tile_rows-1) 的值都置为 0
        for (int r = 0; r < tile_rows; ++r) {
            long long val_idx = val_base + r * tile_cols + col_in_tile;
            values[val_idx] = 0.0f;
        }
    }
}


} // namespace zhao