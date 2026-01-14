#pragma once

// Standard Library
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <stdexcept>

// Thrust Library
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

// CUDA and CUB Library
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

// Custom Library
#include "../helper/sampling_helper.h"



namespace zhao {


namespace cg = cooperative_groups;


/**
 * @brief [Kernel 1] 为每个active window找出Top-K频率的列。
 *  
 * @tparam WARPS_PER_BLOCK 每个Block中的Warp数量，也即批处理大小。
 * @tparam SAMPLE_NUM      每个window最终采样的列数量。
 * @tparam TILE_ROWS       每个tile的行数（编译期常量，需与bcsr.tile_rows一致）。
 * @tparam TILE_COLS       每个tile的列数（编译期常量，需与bcsr.tile_cols一致）。
 *
 * @param d_input_window_offset 输入矩阵的window_offset（设备指针）
 * @param d_input_original_col_indices 输入矩阵的original_col_indices（设备指针）
 * @param d_input_values_condensed 输入矩阵的values_condensed（设备指针）
 * @param d_active_windows GNN的MINI-BATCH中target node所处window的索引。
 * @param d_window_masks GNN的MINI-BATCH中target node所处window内的mask。
 * @param num_active_windows d_active_windows数组的大小。
 * @param d_sampled_col_indices_out 指向中间结果缓冲区的指针，用于存放所有窗口的采样结果。
 * @param d_new_tile_per_window_out 指向中间结果缓冲区的指针，用于存放所有窗口的新tile数量。
 * @param actual_sample_num 实际采样列数量。
 */
template<int WARPS_PER_BLOCK, int SAMPLE_NUM, int TILE_ROWS, int TILE_COLS>
__global__ void warp_centric_find_top_k_kernel(
    const long long* __restrict__ d_input_window_offset,
    const int* __restrict__ d_input_original_col_indices,
    const float* __restrict__ d_input_values_condensed,
    const int* __restrict__ d_active_windows,
    const int* __restrict__ d_window_masks,
    const long long num_active_windows,
    int* __restrict__ d_sampled_col_indices_out,
    int* __restrict__ d_new_tile_per_window_out,
    int actual_sample_num)
{    
    // ==================== 协作组定义 ====================
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // 语义化命名：warp在block中的索引，lane在warp中的索引
    const int warp_id = warp.meta_group_rank();  // 等价于 threadIdx.x / 32
    const int lane_id = warp.thread_rank();       // 等价于 threadIdx.x % 32

    // Block-per-Window：每个block处理一个window
    if (blockIdx.x >= num_active_windows) return;
    
    const int window_id = d_active_windows[blockIdx.x];
    const unsigned int my_mask = (unsigned int)d_window_masks[blockIdx.x];
    const long long tile_start = d_input_window_offset[window_id];
    const long long tile_end = d_input_window_offset[window_id + 1];
    const long long total_tiles = tile_end - tile_start;

    // 空窗口处理
    if (total_tiles == 0 || tile_start < 0 || tile_end < 0) {
        int* window_output = d_sampled_col_indices_out + (long long)blockIdx.x * actual_sample_num;
        for (int i = lane_id; i < actual_sample_num; i += 32) {
            window_output[i] = -1;
        }
        if (threadIdx.x == 0) d_new_tile_per_window_out[blockIdx.x] = 0;
        return;
    }

    // ==================== 参数校验与初始化 ====================
    // 编译期常量计算（确保tile尺寸匹配）
    constexpr int ELEMENTS_PER_TILE = TILE_ROWS * TILE_COLS;    // 16*8
    constexpr int ELEMENTS_PER_THREAD = ELEMENTS_PER_TILE / 32; // 4
    static_assert(ELEMENTS_PER_THREAD * 32 == ELEMENTS_PER_TILE, "TILE尺寸必须是32的倍数");
    
    // 编译期常量计算（确保排序参数匹配）
    constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32;
    constexpr int MAX_ITEMS = SAMPLE_NUM + WARPS_PER_BLOCK *TILE_ROWS * TILE_COLS;  // 最大可能项目数
    constexpr int ITEMS_PER_THREAD_RAW = (MAX_ITEMS + BLOCK_THREADS - 1) / BLOCK_THREADS;  // 每个线程处理的项目数
    // 向上取整到对 CUB 友好的数值 (2的幂次)
    constexpr int ITEMS_PER_THREAD = (ITEMS_PER_THREAD_RAW <= 1) ? 1 : 
                                     (ITEMS_PER_THREAD_RAW <= 2) ? 2 : 
                                     (ITEMS_PER_THREAD_RAW <= 4) ? 4 : 
                                     (ITEMS_PER_THREAD_RAW <= 8) ? 8 : 16;
    // 如果这里报错，说明 Tile 太大或者 SampleNum 太大，导致单 Block 显存/寄存器扛不住了
    static_assert(ITEMS_PER_THREAD >= ITEMS_PER_THREAD_RAW, 
        "Config Error: ITEMS_PER_THREAD capped at 16, but logic requires more!");
    constexpr int GLOBAL_BUFFER_SIZE = ITEMS_PER_THREAD * BLOCK_THREADS;  // 对齐到BLOCK_THREADS的整数倍

    using BlockMergeSort = cub::BlockMergeSort<ColumnCandidate, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockRadixSort = cub::BlockRadixSort<int, BLOCK_THREADS, ITEMS_PER_THREAD>;

    // ==================== 共享内存布局 ====================
    struct __align__(16) SharedStorage {
        union {
            // Phase 1: Warp 统计阶段使用的内存
            struct {
                ColumnCandidate warp_candidates[WARPS_PER_BLOCK][TILE_ROWS * TILE_COLS];
                int warp_candidate_cnt[WARPS_PER_BLOCK];
            } phase1;
            // Phase 2: CUB 排序阶段使用的内存
            typename BlockMergeSort::TempStorage merge_sort_temp;
            typename BlockRadixSort::TempStorage radix_sort_temp;
        } alias;
        
        // 编译器会自动在 alias 和 global_cols 之间填充空字节
        // 确保 global_cols 的地址是 16 的倍数，从而避免对齐问题。
        __align__(16) int global_cols[GLOBAL_BUFFER_SIZE];
        __align__(16) int global_counts[GLOBAL_BUFFER_SIZE];
        __align__(16) int global_candidate_cnt;    
    };

    // 静态声明这个结构体为共享内存
    __shared__ SharedStorage shared;
    
    // ==================== 初始化 ====================
    for (int i = threadIdx.x; i < GLOBAL_BUFFER_SIZE; i += BLOCK_THREADS) {
        shared.global_cols[i] = -1;
        shared.global_counts[i] = INT_MIN; 
    }
    if (block.thread_index().x == 0) {
        shared.global_candidate_cnt = 0;
    }
    
    if (warp_id < WARPS_PER_BLOCK) {
        shared.alias.phase1.warp_candidate_cnt[warp_id] = 0;

        // 使用所有lane并行重置候选条目
        for (int i = lane_id; i < TILE_COLS; i += 32) {
            shared.alias.phase1.warp_candidates[warp_id][i].column_idx = -1;
            shared.alias.phase1.warp_candidates[warp_id][i].frequency = INT_MIN;
        }
    }
    block.sync();
    
    // ==================== 批次处理主循环 ====================
    for (int batch_start = 0; batch_start < total_tiles; batch_start += WARPS_PER_BLOCK) {
        const int batch_tiles = min(WARPS_PER_BLOCK, (int)(total_tiles - batch_start));

        // 重置当前批次的warp候选集和warp的候选集数组
        if (warp_id < WARPS_PER_BLOCK) {
            shared.alias.phase1.warp_candidate_cnt[warp_id] = 0;
            for (int i = lane_id; i < TILE_ROWS * TILE_COLS; i += 32) {
                shared.alias.phase1.warp_candidates[warp_id][i].column_idx = -1;
                shared.alias.phase1.warp_candidates[warp_id][i].frequency = INT_MIN;
            }
        }
        block.sync();
        
        // ==================== Warp内统计（Warp级串行化） ====================
        if (warp_id < batch_tiles) {
            const long long tile_idx = tile_start + batch_start + warp_id;
            
            // 1. 每个线程处理自己的元素
            int my_cols[ELEMENTS_PER_THREAD];
            float my_vals[ELEMENTS_PER_THREAD];

            #pragma unroll
            for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                const int element_offset = lane_id * ELEMENTS_PER_THREAD + e;
                if (element_offset >= ELEMENTS_PER_TILE) {
                    my_cols[e] = -1;  // 无效标记
                    my_vals[e] = 0.0f;
                    continue;
                }
                
                // 计算列索引
                const int col_idx_in_tile = element_offset % TILE_COLS;
                const long long global_col_offset = tile_idx * TILE_COLS + col_idx_in_tile;
                my_cols[e] = d_input_original_col_indices[global_col_offset];

                // 计算值索引并读取值
                const int row_idx_in_tile = element_offset / TILE_COLS;
                const long long global_val_offset = tile_idx * ELEMENTS_PER_TILE + row_idx_in_tile * TILE_COLS + col_idx_in_tile;
                my_vals[e] = d_input_values_condensed[global_val_offset];
            }
            
            // 2. Warp内串行化处理：每个lane依次处理自己的所有新列，只统计非零元素的列
            for (int current_lane = 0; current_lane < 32; ++current_lane) {
                
                // 主导者线程(current_lane)处理自己的所有元素
                if (lane_id == current_lane) {
                    #pragma unroll
                    for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                        int col_to_check = my_cols[e];
                        if (col_to_check == -1) continue;

                        // Masking 检查
                        // 计算当前元素在 16x8 Tile 中属于哪一行
                        // offset = lane_id * 4 + e; row = offset / 8 (即 TILE_COLS)
                        const int row_in_tile = (lane_id * ELEMENTS_PER_THREAD + e) / TILE_COLS;
                        
                        // 检查 Mask 的第 row_in_tile 位是否为 1
                        // 如果为 0，说明这一行不是 Seed，直接跳过计算，不产生噪声邻居
                        // if (!((my_mask >> row_in_tile) & 1u)) {
                        //     continue;
                        // }

                        // 计算混合评分 (Hybrid Score)
                        // 1.0f: 基础拓扑分 (Topology)，保证结构性
                        // 10.0f: 调节因子 (alpha)，控制权重的相关性强度
                        // my_vals[e]: 当前边的权重 (Intensity)
                        // 1000.0f: 将浮点分值放大为整数以便 atomicAdd 累加
                        // fabsf: 取绝对值防止负权重影响（视具体情况可省略）
                        float raw_score = 1.0f + 10.0f * fabsf(my_vals[e]);
                        int score_int = (int)(raw_score * 1000.0f);

                        bool found = false;
                        for (int i = 0; i < shared.alias.phase1.warp_candidate_cnt[warp_id]; ++i) {
                            if (shared.alias.phase1.warp_candidates[warp_id][i].column_idx == col_to_check) {
                                atomicAdd(&shared.alias.phase1.warp_candidates[warp_id][i].frequency, score_int);
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            int idx = atomicAdd(&shared.alias.phase1.warp_candidate_cnt[warp_id], 1);
                            if (idx < TILE_ROWS * TILE_COLS) {
                                shared.alias.phase1.warp_candidates[warp_id][idx].column_idx = col_to_check;
                                shared.alias.phase1.warp_candidates[warp_id][idx].frequency = score_int;
                            }
                        }
                    }
                }
                
                // 在处理完一个lane的所有工作后，才进行一次同步
                // 这确保了下一个lane开始工作时，能看到当前lane的所有更新。
                warp.sync();
            }
        }
        block.sync();
        
        // ==================== Block级别累加合并 ====================
        // 单线程执行合并，避免竞争
        if (block.thread_index().x == 0) {
            for (int w = 0; w < batch_tiles; ++w) {
                const int cnt = shared.alias.phase1.warp_candidate_cnt[w];
                for (int i = 0; i < cnt; ++i) {
                    // 如果全局候选区已满，则提前终止合并
                    if (shared.global_candidate_cnt >= GLOBAL_BUFFER_SIZE) {
                        break; // 退出内层 for 循环
                    }

                    int col_to_add = shared.alias.phase1.warp_candidates[w][i].column_idx;
                    int freq_to_add = shared.alias.phase1.warp_candidates[w][i].frequency;

                    bool found = false;
                    // 在全局列表中查找并累加
                    for (int j = 0; j < shared.global_candidate_cnt; ++j) {
                        if (shared.global_cols[j] == col_to_add) {
                            shared.global_counts[j] += freq_to_add;
                            found = true;
                            break;
                        }
                    }

                    // 如果是新列，并且还有空间，则添加
                    if (!found && shared.global_candidate_cnt < GLOBAL_BUFFER_SIZE) {
                        shared.global_cols[shared.global_candidate_cnt] = col_to_add;
                        shared.global_counts[shared.global_candidate_cnt] = freq_to_add;
                        shared.global_candidate_cnt++;
                    }
                }
                // 如果全局候选区已满，也退出外层循环
                if (shared.global_candidate_cnt >= GLOBAL_BUFFER_SIZE) {
                    break;
                }
            }
        }
        block.sync();
        
        // ==================== Block级别排序 ====================
        // 由整个Block执行排序
        if (block.thread_index().x == 0) {
            shared.global_candidate_cnt = min(GLOBAL_BUFFER_SIZE, shared.global_candidate_cnt);
        }
        block.sync();
        
        if (shared.global_candidate_cnt > 1) {
            // 1. 构造 ColumnCandidate 数组用于排序
            ColumnCandidate thread_candidates[ITEMS_PER_THREAD];
            
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int idx = threadIdx.x * ITEMS_PER_THREAD + i;
                if (idx < shared.global_candidate_cnt) {
                    thread_candidates[i].column_idx = shared.global_cols[idx];
                    thread_candidates[i].frequency = shared.global_counts[idx];
                } else {
                    // 填充无效值，使其在排序后排到最后
                    thread_candidates[i].column_idx = INT_MAX;
                    thread_candidates[i].frequency = 0;
                }
            }
            block.sync();

            // 2. 使用 BlockMergeSort 一次性完成排序
            ColumnCandidateComparator comparator;
            BlockMergeSort(shared.alias.merge_sort_temp).Sort(thread_candidates, comparator);
            
            // 3. 将排序结果写回共享内存
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int idx = threadIdx.x * ITEMS_PER_THREAD + i;
                if (idx < shared.global_candidate_cnt) {
                    shared.global_cols[idx] = thread_candidates[i].column_idx;
                    shared.global_counts[idx] = thread_candidates[i].frequency;
                }
            }
            block.sync();
        }
        block.sync(); // 或许可以删除？
        
        // ==================== 截断到Top-K ====================
        if (block.thread_index().x == 0) {
            shared.global_candidate_cnt = min(SAMPLE_NUM, shared.global_candidate_cnt);
        }
        block.sync();
    }

    // ==================== 按列索引升序排序（最终稳定化输出） ====================
    // 在按频率选出Top-K后，再按列索引从小到大排序以保证确定性输出
    if (shared.global_candidate_cnt > 1) {
        int sort_key[ITEMS_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int idx = threadIdx.x * ITEMS_PER_THREAD + i;
            if (idx < shared.global_candidate_cnt) {
                sort_key[i] = shared.global_cols[idx];
            } else {
                sort_key[i] = INT_MAX;
            }
        }

        block.sync();

        // 执行块内升序排序
        BlockRadixSort(shared.alias.radix_sort_temp).Sort(sort_key);
        
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int idx = threadIdx.x * ITEMS_PER_THREAD + i;
            if (idx < shared.global_candidate_cnt) {
                shared.global_cols[idx] = sort_key[i];
            }
        }
        block.sync();
    }
    block.sync(); // 或许可以删除？
    
    // ==================== 写回全局内存 ====================
    if (warp_id == 0) {
        int* window_output = d_sampled_col_indices_out + (long long)blockIdx.x * actual_sample_num;
        const int valid_cnt = min(shared.global_candidate_cnt, actual_sample_num);
        int new_tiles = (valid_cnt + TILE_COLS - 1) / TILE_COLS;

        // 写入新 Tile 数
        if (lane_id == 0) {
            d_new_tile_per_window_out[blockIdx.x] = new_tiles;
        }
        
        // 并行写回有效候选
        for (int i = lane_id; i < valid_cnt; i += 32) {
            window_output[i] = shared.global_cols[i];
        }
        
        // 并行填充-1
        for (int i = valid_cnt + lane_id; i < actual_sample_num; i += 32) {
            window_output[i] = -1;
        }
    }
}


/**
 * @brief [Kernel 2] 重构BCSR矩阵（O(1)值查找版）
 * 
 * @tparam WARPS_PER_BLOCK 每个Block的Warp数量（推荐8-16）
 * @tparam SAMPLE_NUM 最大采样列数（编译期常量）
 * @tparam TILE_ROWS 每tile行数（编译期常量）
 * @tparam TILE_COLS 每tile列数（编译期常量）
 *
 * @param d_input_num_entries 输入矩阵的非零元总数
 * @param d_input_window_offset 输入矩阵的window_offset（设备指针）
 * @param d_input_values_condensed 输入矩阵的values_condensed（设备指针）
 * @param d_active_windows GNN的MINI-BATCH中target node所处window的索引。
 * @param d_sampled_col_indices 采样列数组（num_windows × SAMPLE_NUM）
 * @param d_col_to_val_map 列到值的映射表（num_windows × SAMPLE_NUM）
 * @param d_output_num_windows 输出矩阵的窗口数
 * @param d_output_window_offset 输出矩阵的window_offset（设备指针）
 * @param d_output_original_col_indices 输出矩阵的original_col_indices（设备指针）输出原始列索引
 * @param d_output_values_condensed 输出矩阵的values_condensed（设备指针）
 * @param actual_sample_num 实际采样列数量。
 */
template<int WARPS_PER_BLOCK, int SAMPLE_NUM, int TILE_ROWS, int TILE_COLS>
__global__ void recompress_sampled_bcsr_kernel(
    const long long d_input_num_entries,
    const long long* __restrict__ d_input_window_offset,
    const float* __restrict__ d_input_values_condensed,
    const int* __restrict__ d_active_windows,
    const int* __restrict__ d_sampled_col_indices,
    const int* __restrict__ d_col_to_val_map,
    const long long d_output_num_windows,
    const long long* __restrict__ d_output_window_offset,
    int* __restrict__ d_output_original_col_indices,
    float* __restrict__ d_output_values_condensed,
    int actual_sample_num)
{
    // batch_idx 对应输出 buffer 的下标 (0, 1, 2...)
    const int batch_idx = blockIdx.x;
    if (batch_idx >= d_output_num_windows) return;

    // global_window_id 对应原始大图的行号 (用于读取输入)
    const int global_window_id = d_active_windows[batch_idx];
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    const int warp_id = warp.meta_group_rank();
    const int lane_id = warp.thread_rank();

    constexpr int ELEMENTS_PER_THREAD = 4;

    // ============= 静态共享内存 =============
    __shared__ int s_sampled_cols[SAMPLE_NUM];
    __shared__ int s_col_to_val_map[SAMPLE_NUM];

    // 协作加载当前window的采样列和映射表
    const long long base_offset = (long long)batch_idx * actual_sample_num;
    for (int i = threadIdx.x; i < SAMPLE_NUM; i += block.size()) {
        if (i < actual_sample_num) {
            // 有效范围内：读取 Global Memory
            s_sampled_cols[i] = d_sampled_col_indices[base_offset + i];
            s_col_to_val_map[i] = d_col_to_val_map[base_offset + i];
        } else {
            // 超出实际采样数，但在模板容量内：填充 -1 (Padding)
            // 这一步对于后续逻辑的安全至关重要
            s_sampled_cols[i] = -1;
            s_col_to_val_map[i] = -1;
        }
    }
    __syncthreads();

    // 计算当前窗口在输入values数组中的起始位置
    const long long input_tile_start = d_input_window_offset[global_window_id];
    const long long input_val_base = input_tile_start * TILE_ROWS * TILE_COLS;  // 窗口的值基地址

    // ============= 计算输出范围 =============
    const long long output_tile_start = d_output_window_offset[batch_idx];
    const long long output_tile_end = d_output_window_offset[batch_idx + 1];
    const long long output_tile_count = output_tile_end - output_tile_start;

    // ============= Warp并行处理tile =============
    // 每个warp处理1个输出tile
    for (int tile_offset = warp_id; tile_offset < output_tile_count; tile_offset += WARPS_PER_BLOCK) {        
        const long long output_tile_idx = output_tile_start + tile_offset;
        const long long sampled_col_start = tile_offset * TILE_COLS;

        // ---------- 填充original_col_indices（前TILE_COLS个threads）----------
        if (lane_id < TILE_COLS) {
            const long long sampled_idx = sampled_col_start + lane_id;
            int col_value = -1;
            if (sampled_idx < actual_sample_num) {
                col_value = s_sampled_cols[sampled_idx];
            }

            d_output_original_col_indices[output_tile_idx * TILE_COLS + lane_id] = col_value;
        }

        // ---------- 填充values_condensed（所有threads协作）----------
        // 每个thread处理4个连续元素（thread0: [0-3], thread1: [4-7]...）
        #pragma unroll ELEMENTS_PER_THREAD
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int elem_idx = lane_id * ELEMENTS_PER_THREAD + i;
            if (elem_idx >= TILE_ROWS * TILE_COLS) continue;
            
            const int row_offset = elem_idx / TILE_COLS;
            const int col_offset = elem_idx % TILE_COLS;
            const long long sampled_idx = sampled_col_start + col_offset;
            
            float value = 0.0f;
            if (sampled_idx < actual_sample_num && s_sampled_cols[sampled_idx] != -1) {
                // O(1)查找：直接获取映射信息
                const int packed_val = s_col_to_val_map[sampled_idx];
                if (packed_val != -1) {
                    // 解压 (Unpacking)
                    const int map_tile_idx = packed_val >> 4;   // 高位是 tile index
                    const int map_col_offset = packed_val & 0xF; // 低4位是 col offset

                    // 使用相对tile索引计算全局值索引
                    const long long tile_val_base = input_val_base + (long long)map_tile_idx * TILE_ROWS * TILE_COLS;
                    const long long input_val_idx = tile_val_base + row_offset * TILE_COLS + map_col_offset;
                    if (input_val_idx < d_input_num_entries) {
                        value = d_input_values_condensed[input_val_idx];
                    }
                }
            }
            
            const long long output_idx = output_tile_idx * TILE_ROWS * TILE_COLS + elem_idx;
            d_output_values_condensed[output_idx] = value;
        }
    }
}


/**
 * @brief [Helper Kernel 1] 统计每个Window中实际有效的采样边数 (非 -1 的数量)
 * 
 * @param d_sampled_cols 输入：采样后的列索引矩阵 [num_windows * sample_num]
 * @param num_windows 窗口数量
 * @param sample_num 每个窗口的采样数 (stride)
 * @param d_valid_counts_out 输出：每个窗口的有效边数 [num_windows]
 */
static __global__ void count_valid_edges_kernel(
    const int* __restrict__ d_sampled_cols,
    const int num_windows,
    const int sample_num,
    int* __restrict__ d_valid_counts_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_windows) return;

    int count = 0;
    const int row_start = idx * sample_num;
    
    // 遍历当前窗口的所有采样槽位
    for (int i = 0; i < sample_num; ++i) {
        if (d_sampled_cols[row_start + i] != -1) {
            count++;
        }
    }
    d_valid_counts_out[idx] = count;
}


/**
 * @brief [Helper Kernel 2] 完成：提取 + 转置坐标 + 读取原始值
 * 
 * @param d_sampled_cols 采样后的列索引矩阵
 * @param d_col_to_val_map 列到值的映射索引
 * @param d_active_windows 当前活跃的窗口 ID
 * @param d_write_offsets 由 count_valid_edges_kernel + Scan 计算出的写入起始位置
 * @param d_orig_win_off  原图 window_offset (用于定位原始值)
 * @param d_orig_values   原图 values (用于读取原始值)
 * @param num_windows 窗口数量
 * @param sample_num 每个窗口的采样数 (stride)
 * @param tile_rows 每个tile的行数。
 * @param tile_cols 每个tile的列数。
 * @param d_coo_rows_out 输出：转置后的行 (即原图的列)
 * @param d_coo_cols_out 输出：转置后的列 (即原图的行)
 * @param d_coo_vals_out 输出：值
 */
static __global__ void extract_transposed_coo_and_val_kernel(
    const int* __restrict__ d_sampled_cols,
    const int* __restrict__ d_col_to_val_map,
    const int* __restrict__ d_active_windows,
    const int* __restrict__ d_write_offsets,
    const long long* __restrict__ d_orig_win_off, 
    const float* __restrict__ d_orig_values,     
    const int num_windows,
    const int sample_num,
    const int tile_rows,
    const int tile_cols,
    int* __restrict__ d_coo_rows_out,
    int* __restrict__ d_coo_cols_out,
    float* __restrict__ d_coo_vals_out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_windows) return;

    // 获取当前窗口的全局 ID (对应原图的 Block Row ID)
    const int global_window_id = d_active_windows[idx];
    
    // 获取当前窗口在原图中的起始 Tile 索引
    const long long orig_win_start_tile = d_orig_win_off[global_window_id];

    int write_pos = d_write_offsets[idx];
    const int row_read_start = idx * sample_num;

    for (int i = 0; i < sample_num; ++i) {
        int col_global = d_sampled_cols[row_read_start + i];
        
        if (col_global != -1) {
            // 解析映射信息
            int map_val = d_col_to_val_map[row_read_start + i];
            if (map_val == -1) continue;
            // Packed format: (relative_tile_idx << 4) | col_offset_in_tile
            int relative_tile_idx = map_val >> 4;
            int col_offset_in_tile = map_val & 0xF;

            // 定位到原图具体的 Tile
            long long target_tile_idx = orig_win_start_tile + relative_tile_idx;
            
            // 遍历 Tile 的 16 行 (因为 BCSR 采样是 Block 级的，整列 Tile 都被选中)
            long long val_base_addr = target_tile_idx * tile_rows * tile_cols;

            for (int r = 0; r < tile_rows; ++r) {
                // 读取原始值
                float val = d_orig_values[val_base_addr + r * tile_cols + col_offset_in_tile];

                // 执行转置输出
                // A^T Row = 原图 Col (Sampled Global Node ID)
                // 这个保持不变，因为 Grad_In 需要散射回全局或由 Global ID 索引
                d_coo_rows_out[write_pos] = col_global; 
                
                // 使用 idx (当前 Batch 的 Window 下标, 0..batch_size-1)
                // 这样生成的列索引才能正确对应 Grad_Out (Dense Matrix) 的行
                d_coo_cols_out[write_pos] = idx * tile_rows + r;
                
                d_coo_vals_out[write_pos] = val;
                write_pos++; // 移动写入指针
            }
        }
    }
}


/**
 * @brief [Helper Kernel 3] 构建列到值的映射表
 * 
 * @param window_offset 原图 window_offset (用于定位原始值)
 * @param original_col_indices 原图 original_col_indices (用于读取原始值)
 * @param active_windows GNN的MINI-BATCH中target node所处window的索引。
 * @param sampled_cols 采样后的列索引矩阵 [num_windows * sample_num]
 * @param col_to_val_map 列到值的映射索引 [num_windows * sample_num]
 * @param num_active_windows 活跃的窗口数量
 * @param sample_num 每个窗口的采样数 (stride)
 * @param tile_rows 每个tile的行数。
 * @param tile_cols 每个tile的列数。
 */
static __global__ void build_col_to_val_map_kernel(
    const long long* __restrict__ window_offset,
    const int* __restrict__ original_col_indices,
    const int* __restrict__ active_windows,
    const int* __restrict__ sampled_cols,
    int* __restrict__ col_to_val_map,
    int num_active_windows,
    int sample_num,
    int tile_rows,
    int tile_cols)
{
    // 1. 计算当前线程负责的索引 (Thread-level parallelism)
    // 每个线程负责一个 (window, sample_slot)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active_windows * sample_num) return;

    // 2. 解析 Window 索引
    // idx 对应 flat 后的数组，需要还原出是第几个 active window
    int w_idx = idx / sample_num;
    // int s_idx = idx % sample_num; // [Deleted] 移除此行以消除 unused variable 警告

    // 3. 读取目标列 (Target Column)
    int target_col = sampled_cols[idx];

    // 4. 处理 Padding (-1)
    // 与 CPU 逻辑保持一致：如果是占位符，直接填 -1 并返回
    if (target_col == -1) {
        col_to_val_map[idx] = -1;
        return;
    }

    // 5. 获取 Window 的物理范围
    int global_window_id = active_windows[w_idx];
    // 利用 __ldg 显式告诉编译器这只读且走 Read-Only Cache (虽然编译器通常会自动优化)
    long long tile_start = __ldg(&window_offset[global_window_id]);
    long long tile_end   = __ldg(&window_offset[global_window_id + 1]);

    // 6. 线性搜索 (Linear Search)
    // 在 GPU 上，由于线程极多，这种短线性搜索效率很高
    int result = -1;

    for (long long t = tile_start; t < tile_end; ++t) {
        long long col_base = t * tile_cols;
        
        // 内层循环：在 Tile 内部查找
        // 针对 tile_cols=8 进行循环展开优化
        #pragma unroll
        for (int c = 0; c < 8; ++c) { 
            // 注意：如果你的 tile_cols 可能是其他值，请将 8 改为 tile_cols 或去掉 unroll
            if (original_col_indices[col_base + c] == target_col) {
                // 找到目标！
                long long relative_tile_idx = t - tile_start;
                
                // 7. 位压缩 (Bit Packing) - 逻辑与 CPU 代码完全一致
                // Format: (relative_tile_idx << 4) | col_offset
                result = (int)((relative_tile_idx << 4) | c);
                
                // GPU 上没有简单的 break out of nested loops，使用 goto 是最高效的跳出方式
                goto FOUND;
            }
        }
    }

FOUND:
    // 8. 写回结果
    col_to_val_map[idx] = result;
}


}   // namespace zhao
