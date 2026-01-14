#pragma once

// Standard Library
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <cstdint>

// CUDA and CUB Library
#include <cuda_runtime.h>
#include <mma.h>
#include <cooperative_groups.h>

// custom config header file
#include "../helper/config.h"


namespace zhao {


namespace cg = cooperative_groups;


/**
 * @brief Tensor Core SpMM Kernel (Base Version)
 * 计算 C = A * B
 * A: BCSR 稀疏矩阵 (M x K)
 * B: 稠密矩阵 (K x N)
 * C: 稠密矩阵 (M x N)
 *
 * @param N 矩阵 B/C 的列数
 * @param window_offset 窗口偏移量（设备指针）
 * @param original_col_indices 列索引
 * @param values_condensed 非零值
 * @param input_B 输入稠密矩阵
 * @param output_C 输出稠密矩阵
 */
static __global__ void tcspmm_base_kernel(
    long long N,
    const long long *__restrict__ window_offset, 
    const int *__restrict__ original_col_indices,
    const float *__restrict__ values_condensed,
    const float *__restrict__ input_B, 
    float *output_C)
{
    // 初始化 Cooperative Groups
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Block 处理一个 Window (行块)
    // 这里的 blockIdx.x 依然对应 Window ID
    const unsigned bid = block.group_index().x; // 等价于 blockIdx.x
    
    // 使用 long long 防止大图索引溢出
    long long block_start = window_offset[bid];
    long long block_end = window_offset[bid + 1];
    
    if(block_start == block_end) return;

    // Warp 内部分工
    // 使用 CG API 获取 Warp ID 和 Lane ID
    // meta_group_rank(): 当前 Warp 在 Block 中的 ID (等价于 threadIdx.y)
    const unsigned wid = warp.meta_group_rank();      
    // thread_rank(): 当前线程在 Warp 中的 ID (等价于 threadIdx.x)
    const unsigned laneid = warp.thread_rank();   

    // warp_offset: 当前 Warp 负责计算 N 维度上的哪一部分
    const unsigned warp_offset = wid * tile_N; 

    const unsigned groupID = laneid >> 2;
    const unsigned threadID_in_group = laneid % 4;
    
    // Tensor Core 片段 (Fragments)
    uint32_t RA[4]; 
    uint32_t RB[2]; 
    float RC[4] = {0.0f}; 
    
    // --------------------------------------------------------
    // Fragment 索引映射 (根据 m16n8k8 布局)
    // --------------------------------------------------------
    const unsigned row_a0 = groupID;
    const unsigned row_a1 = groupID + 8;
    const unsigned row_a2 = groupID;
    const unsigned row_a3 = groupID + 8;

    const unsigned col_a0 = threadID_in_group;
    const unsigned col_a1 = threadID_in_group;
    const unsigned col_a2 = threadID_in_group + 4;
    const unsigned col_a3 = threadID_in_group + 4;

    const unsigned row_b0 = threadID_in_group;
    const unsigned row_b1 = threadID_in_group + 4;

    const unsigned col_b0 = groupID;
    const unsigned col_b1 = groupID;

    // --------------------------------------------------------
    // 计算 A 矩阵 Tile 内的相对索引
    // --------------------------------------------------------
    const unsigned A_index_0 = row_a0 * tile_K + col_a0;
    const unsigned A_index_1 = row_a1 * tile_K + col_a1;
    const unsigned A_index_2 = row_a2 * tile_K + col_a2;
    const unsigned A_index_3 = row_a3 * tile_K + col_a3;

    long long A_offset, B_offset;
    long long dense_B_idx_0, dense_B_idx_1;

    // --------------------------------------------------------
    // 主循环：遍历当前 Window 中的所有 Tile (沿着 K 维度)
    // --------------------------------------------------------
    for (long long i = block_start; i < block_end; i++) {

        // 1. 加载稀疏矩阵 A 的左侧 Tile (values_condensed)
        A_offset = i * tile_M * tile_K; 
        
        cvt_F32_to_TF32(RA[0], values_condensed[A_offset + A_index_0]);
        cvt_F32_to_TF32(RA[1], values_condensed[A_offset + A_index_1]);
        cvt_F32_to_TF32(RA[2], values_condensed[A_offset + A_index_2]);
        cvt_F32_to_TF32(RA[3], values_condensed[A_offset + A_index_3]);

        // 2. 加载稠密矩阵 B 的右侧 Tile
        B_offset = i * tile_K;

        // 先读取列索引，检查是否有效
        int col_idx_val_0 = original_col_indices[B_offset + row_b0];
        int col_idx_val_1 = original_col_indices[B_offset + row_b1];

        // 默认值为 0 (对应 padding 部分的乘法贡献应为 0)
        float val_B_0 = 0.0f;
        if (col_idx_val_0 != -1) {
            dense_B_idx_0 = (long long)col_idx_val_0 * N + col_b0 + warp_offset;
            val_B_0 = input_B[dense_B_idx_0];
        }

        float val_B_1 = 0.0f;
        if (col_idx_val_1 != -1) {
            dense_B_idx_1 = (long long)col_idx_val_1 * N + col_b1 + warp_offset;
            val_B_1 = input_B[dense_B_idx_1];
        }
        
        // 将安全获取的值转换为 TF32 格式
        cvt_F32_to_TF32(RB[0], val_B_0);
        cvt_F32_to_TF32(RB[1], val_B_1);

        // 显式同步 (可选，MMA指令本身隐含同步，但加上更符合逻辑)
        // warp.sync(); 

        // 3. Tensor Core 计算乘累加 (MMA)
        HMMA1688(RC[0], RC[1], RC[2], RC[3], 
                 RA[0], RA[1], RA[2], RA[3],
                 RB[0], RB[1]);
    }
    
    // --------------------------------------------------------
    // 结果写回 (Write Back)
    // --------------------------------------------------------
    const unsigned row_c0 = groupID;
    const unsigned row_c1 = groupID;
    const unsigned row_c2 = groupID + 8;
    const unsigned row_c3 = groupID + 8;

    const unsigned col_c0 = (threadID_in_group * 2) + (0 & 0x1);
    const unsigned col_c1 = (threadID_in_group * 2) + (1 & 0x1);
    const unsigned col_c2 = (threadID_in_group * 2) + (2 & 0x1);
    const unsigned col_c3 = (threadID_in_group * 2) + (3 & 0x1);

    // 计算 C 的输出位置
    long long C_offset = (long long)bid * tile_M * N + warp_offset;

    // 添加边界检查
    // 只有当列索引小于 N 时才写入

    if (warp_offset + col_c0 < N) {
        const long long C_index_0 = C_offset + row_c0 * N + col_c0;
        output_C[C_index_0] = RC[0];
    }

    if (warp_offset + col_c1 < N) {
        const long long C_index_1 = C_offset + row_c1 * N + col_c1;
        output_C[C_index_1] = RC[1];
    }

    if (warp_offset + col_c2 < N) {
        const long long C_index_2 = C_offset + row_c2 * N + col_c2;
        output_C[C_index_2] = RC[2];
    }

    if (warp_offset + col_c3 < N) {
        const long long C_index_3 = C_offset + row_c3 * N + col_c3;
        output_C[C_index_3] = RC[3];
    }
    
}


/**
 * @brief Atomic Transpose SpMM Kernel
 * 逻辑：遍历 Forward 图 A，对于每条边 (row, col, val)，执行 grad_in[col] += val * grad_out[row]
 * 线程策略：一个 Block 处理 A 的一个 Window (16行)
 *          Block.x = Feature_Dim (并行处理特征维度)
 * @param num_windows A 的 Window 数
 * @param feat_dim 特征维度 K
 * @param window_offset Window 指针
 * @param col_indices 列索引
 * @param values 值
 * @param grad_out 输入梯度 [num_rows, K]
 * @param grad_in 输出梯度 [num_cols, K] (需预先置0)
 */
template <int TILE_ROWS, int TILE_COLS>
__global__ void spmm_bcsr_transpose_atomic_kernel(
    const long long num_windows,
    const int feat_dim,
    const long long* __restrict__ window_offset,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in)
{
    // 每个 Block 处理一个 Window
    const int win_id = blockIdx.x;
    if (win_id >= num_windows) return;

    // 当前线程负责的 feature index
    const int f = threadIdx.x;
    // 如果 feat_dim 很大，这里应该加循环；假设 feat_dim <= blockDim.x
    if (f >= feat_dim) return;

    // Window 的起始/结束 Tile 索引
    const long long tile_start = window_offset[win_id];
    const long long tile_end = window_offset[win_id + 1];

    // 全局行偏移
    const int row_base = win_id * TILE_ROWS;

    // 遍历该 Window 下所有的 Tile
    for (long long t = tile_start; t < tile_end; ++t) {
        // 每个 Tile 有 TILE_COLS (8) 列
        const int* tile_cols_ptr = col_indices + t * TILE_COLS;
        const float* tile_vals_ptr = values + t * TILE_ROWS * TILE_COLS;

        // 遍历 Tile 的 16 行
        for (int r = 0; r < TILE_ROWS; ++r) {
            // 读取当前行的 grad_out (Coalesced Read: 线程 f 读取 feature f)
            // grad_out layout: Row-Major [GlobalRow, Feat]
            float g_val = grad_out[(row_base + r) * feat_dim + f];

            // 稀疏优化：如果梯度极小，跳过
            if (abs(g_val) < 1e-9) continue;

            // 遍历 Tile 的 8 列
            for (int c = 0; c < TILE_COLS; ++c) {
                int col_idx = tile_cols_ptr[c];
                
                // 跳过 Padding (-1)
                if (col_idx == -1) continue;

                // 读取边权重 A[r, c]
                float weight = tile_vals_ptr[r * TILE_COLS + c];

                // Atomic Add 到 grad_in
                // grad_in layout: [GlobalCol, Feat]
                if (weight != 0.0f) {
                    atomicAdd(&grad_in[col_idx * feat_dim + f], weight * g_val);
                }
            }
        }
    }
}


} // namespace zhao