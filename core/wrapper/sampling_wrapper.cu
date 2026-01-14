// PyTorch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// standard library
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <tuple>

// CUDA library
#include <cuda_runtime.h>
#include <omp.h> 

// Thrust library
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>

// Custom Headers
#include "../helper/macros.h"
#include "../helper/timer.h"
#include "../helper/config.h"
#include "../include/sampling.cuh"

using namespace zhao;



/**
 * @brief 内部静态函数：执行单层采样核心逻辑
 * 
 * @param window_offset 输入BCSR矩阵的window_offset（GPU指针）
 * @param original_col_indices 输入BCSR矩阵的original_col_indices（GPU指针）
 * @param values_condensed 输入BCSR矩阵的values_condensed（GPU指针）
 * @param active_windows GNN的MINI-BATCH中target node所处window的索引。
 * @param window_masks GNN的MINI-BATCH中target node所处window内的mask。
 * @param fanout 每个target node的采样列数量。
 * @param tile_rows 每个tile的行数。
 * @param tile_cols 每个tile的列数。
 * @param warps_per_block 每个block的warp数量。
 * @param stream CUDA流。
 * 
 * @return 一个pair: <std::vector<torch::Tensor>, SamplingTimingResult>
 * @return 第一个元素是包含三个张量的向量：out_window_offset, out_original_col_indices, out_values_condensed
 * @return 第二个元素是采样时间信息
 */
static std::pair<std::vector<torch::Tensor>, SamplingTimingResult> sample_one_layer_internal(
    const torch::Tensor& window_offset,
    const torch::Tensor& original_col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& active_windows,
    const torch::Tensor& window_masks,
    int fanout,
    int tile_rows,
    int tile_cols,
    int warps_per_block,
    cudaStream_t stream)
{
    SamplingTimingResult timer_result;
    CUDATimer total_timer;
    total_timer.start();
    
    // 基础参数检查已经在外部做过，这里略过
    long long num_active_windows = active_windows.size(0);
    // 确保 fanout 对齐到 32 (Kernel Warp 限制)
    int sample_num = ((fanout + 31) / 32) * 32; 

    auto opts = active_windows.options().dtype(torch::kInt32); // 复用 device
    auto sampled_cols = torch::full({num_active_windows * sample_num}, -1, opts);
    auto tile_counts_per_window = torch::zeros({num_active_windows}, opts);

    // === Phase 1: Warp-Centric Sampling ===
    {
        CUDATimer p1_timer;
        p1_timer.start();
        
        dim3 grid(num_active_windows);
        dim3 block(32 * warps_per_block);
        
        // 获取指针
        const long long* d_win = GET_CONST_PTR(long long, window_offset);
        const int* d_col = GET_CONST_PTR(int, original_col_indices);
        const float* d_val = GET_CONST_PTR(float, values_condensed);
        const int* d_act = GET_CONST_PTR(int, active_windows);
        const int* d_mask = GET_CONST_PTR(int, window_masks);
        int* d_out_cols = GET_PTR(int, sampled_cols);
        int* d_out_tiles = GET_PTR(int, tile_counts_per_window);

        if (tile_rows == 16 && tile_cols == 8) {
            // 使用 macros.h 中的 DISPATCH 宏
            // 宏将自动根据 fanout (ACTUAL) 选择最合适的 CAPACITY
            if (warps_per_block == 4) {
                DISPATCH_SAMPLING(
                    4, sample_num, stream,
                    d_win, d_col, d_val,
                    d_act, d_mask, num_active_windows,
                    d_out_cols, d_out_tiles
                );
            } else if (warps_per_block == 8) {
                DISPATCH_SAMPLING(
                    8, sample_num, stream,
                    d_win, d_col, d_val,
                    d_act, d_mask, num_active_windows,
                    d_out_cols, d_out_tiles
                );
            } else {
                TORCH_CHECK(false, "Unsupported warps_per_block: ", warps_per_block, ". Only 4, 8, 16 allowed.");
            }
        } else {
            TORCH_CHECK(false, "Currently only supports tile 16x8");
        }
        // 注意：CHECK_CUDA 会隐式同步，如果为了极致精确计时，可暂时移除或保留
        // CHECK_CUDA(cudaGetLastError());

        p1_timer.stop();
        timer_result.phase1_sampling_time = p1_timer.elapsed();
    }

    // === Phase 2: Scan: 计算 Prefix Sum 得到子图的总 Tile 数 ===
    auto out_window_offset = torch::empty(
        {num_active_windows + 1}, 
        window_offset.options().device(active_windows.device()) // 继承 dtype (Int64), 但强制使用 GPU
    );
    out_window_offset[0].zero_();
    
    int* d_counts = GET_PTR(int, tile_counts_per_window);
    long long* d_out_off = GET_PTR(long long, out_window_offset);
    
    thrust::inclusive_scan(
        thrust::cuda::par.on(stream),
        d_counts,
        d_counts + num_active_windows,
        d_out_off + 1
    );

    // 新代码：直接在 CPU 计算理论最大值 (Zero Overhead)
    // 每个 Window 最多产生的 Tile 数 = ceil(fanout / tile_cols)
    // 注意：这里用 fanout (实际采样数)，而不是 sample_num (对齐后的)
    int max_tiles_per_window = (sample_num + tile_cols - 1) / tile_cols;
    long long max_total_tiles = (long long)num_active_windows * max_tiles_per_window;

    // === Phase 3: Mapping ===
    auto col_to_val_map = torch::empty({num_active_windows * sample_num}, opts);
    {
        CUDATimer p2_timer;
        p2_timer.start();

        int threads = 256;
        int blocks = (num_active_windows * sample_num + threads - 1) / threads;
        
        build_col_to_val_map_kernel<<<blocks, threads, 0, stream>>>(
            GET_CONST_PTR(long long, window_offset),
            GET_CONST_PTR(int, original_col_indices),
            GET_CONST_PTR(int, active_windows),
            GET_PTR(int, sampled_cols),
            GET_PTR(int, col_to_val_map),
            num_active_windows, sample_num, tile_rows, tile_cols
        );
        // CHECK_CUDA(cudaGetLastError());

        p2_timer.stop();
        timer_result.phase2_mapping_time = p2_timer.elapsed();
    }

    // === Phase 4: Recompression ===
    // 显存预分配
    // 使用估算的 max_total_tiles 分配显存
    // 注意：这样分配的 Tensor 会比实际数据大（尾部会有 Padding），
    // 但因为后续 SpMM/Reindex 都是严格按照 window_offset 遍历的，所以是安全的。
    auto out_original_col_indices = torch::full(
        {max_total_tiles * tile_cols}, -1, 
        original_col_indices.options().device(active_windows.device())
    );
    auto out_values_condensed = torch::full(
        {max_total_tiles * tile_rows * tile_cols}, -1.0f, 
        values_condensed.options().device(active_windows.device())
    );
    {
        CUDATimer p3_timer;
        p3_timer.start();

        const float* d_val= GET_CONST_PTR(float, values_condensed);
        const long long* d_win = GET_CONST_PTR(long long, window_offset);
        const int* d_act = GET_CONST_PTR(int, active_windows);
        const int* d_spl_col = GET_PTR(int, sampled_cols);
        const int* d_c2v_map = GET_PTR(int, col_to_val_map);
        const long long* d_out_win = GET_PTR(long long, out_window_offset);
        int* d_out_col = GET_PTR(int, out_original_col_indices);
        float* d_out_val = GET_PTR(float, out_values_condensed);

        dim3 grid(num_active_windows);
        dim3 block(32 * warps_per_block);
        
        // 模板分发 (需匹配 Phase 1)
        if (tile_rows == 16 && tile_cols == 8) {
            // 使用 macros.h 中的 DISPATCH 宏
            if (warps_per_block == 4) {
                DISPATCH_RECOMPRESS(
                    4, sample_num, stream,
                    values_condensed.numel(),
                    d_win, d_val, d_act,
                    d_spl_col, d_c2v_map,
                    num_active_windows,
                    d_out_win, d_out_col, d_out_val
                );
            } else if (warps_per_block == 8) {
                DISPATCH_RECOMPRESS(
                    8, sample_num, stream,
                    values_condensed.numel(),
                    d_win, d_val, d_act,
                    d_spl_col, d_c2v_map,
                    num_active_windows,
                    d_out_win, d_out_col, d_out_val
                );
            } else {
                TORCH_CHECK(false, "Unsupported warps_per_block: ", warps_per_block, ". Only 4, 8, 16 allowed.");
            }
        }
        //CHECK_CUDA(cudaGetLastError());

        p3_timer.stop();
        timer_result.phase3_recompression_time = p3_timer.elapsed();
    }

    total_timer.stop();
    timer_result.total_time = total_timer.elapsed();

    // 返回：Forward 三件套 + 计时
    return {{out_window_offset, out_original_col_indices, out_values_condensed}, timer_result};
}


/**
 * @brief 根据种子节点计算 Active Windows 和对应的 Masks
 *
 * @param seeds_raw 上一层采样的列索引
 * @param tile_rows 每个tile的行数。
 * @param stream CUDA流。
 * 
 * @return 一个 pair: {active_windows, window_masks}
 */
std::pair<torch::Tensor, torch::Tensor> prepare_windows_and_masks(
    const torch::Tensor& seeds_raw,
    int tile_rows,
    cudaStream_t stream)
{
    // 1. 数据准备
    torch::Tensor seeds = seeds_raw.to(torch::kInt32);
    int num_seeds = seeds.numel();
    auto opts = seeds.options().dtype(torch::kInt32);

    if (num_seeds == 0) {
        return {torch::empty({0}, opts), torch::empty({0}, opts)};
    }

    // 临时 Buffer：存放每个 seed 对应的 window_id 和 single_bit_mask
    auto window_ids_tmp = torch::empty({num_seeds}, opts);
    auto bitmasks_tmp   = torch::empty({num_seeds}, opts);

    int* d_in = GET_PTR(int, seeds);
    int* d_wins_tmp = GET_PTR(int, window_ids_tmp);
    unsigned int* d_masks_tmp = (unsigned int*)GET_PTR(int, bitmasks_tmp);

    auto policy = thrust::cuda::par.on(stream);

    // 2. Transform: 将 NodeID 映射为 (WindowID, BitMask)
    // 逻辑：node_idx -> {node_idx / 16, 1 << (node_idx % 16)}
    thrust::transform(policy, d_in, d_in + num_seeds, 
        thrust::make_zip_iterator(thrust::make_tuple(d_wins_tmp, d_masks_tmp)),
        [=] __device__ (int val) {
            if (val == -1) return thrust::make_tuple(INT_MAX, 0u);
            return thrust::make_tuple(val / tile_rows, 1u << (val % tile_rows));
        }
    );

    // 3. Sort: 按 WindowID 排序 (因为 Python 端已排过序，这里几乎是瞬时的)
    thrust::sort_by_key(policy, d_wins_tmp, d_wins_tmp + num_seeds, d_masks_tmp);

    // 4. ReduceByKey: 聚合相同 Window 的 Mask (执行位或 OR 运算)
    auto out_windows = torch::empty({num_seeds}, opts);
    auto out_masks   = torch::empty({num_seeds}, opts);
    int* d_out_wins = GET_PTR(int, out_windows);
    unsigned int* d_out_masks = (unsigned int*)GET_PTR(int, out_masks);

    auto end_pair = thrust::reduce_by_key(
        policy,
        d_wins_tmp, d_wins_tmp + num_seeds, // Keys
        d_masks_tmp,                        // Values
        d_out_wins,                         // Result Keys
        d_out_masks,                        // Result Aggregated Values
        thrust::equal_to<int>(),            // 判断 Key 是否相同
        thrust::bit_or<unsigned int>()      // 相同 Key 的 Value 执行位或
    );

    int num_active = end_pair.first - d_out_wins;
    
    // 剔除无效的 INT_MAX (由 -1 产生的)
    if (num_active > 0) {
        int last_val;
        cudaMemcpyAsync(&last_val, d_out_wins + num_active - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (last_val == INT_MAX) num_active--;
    }

    return {out_windows.slice(0, 0, num_active), out_masks.slice(0, 0, num_active)};
}


/**
 * @brief 多层采样主入口 (C++ Loop)
 * 
 * @param window_offset 输入BCSR矩阵的window_offset（GPU指针）
 * @param original_col_indices 输入BCSR矩阵的original_col_indices（GPU指针）
 * @param values_condensed 输入BCSR矩阵的values_condensed（GPU指针）
 * @param initial_seeds Layer 0 的种子节点
 * @param fanouts 采样扇出
 * @param num_global_nodes 接收真实的全图节点数
 * @param tile_rows 每个tile的行数。
 * @param tile_cols 每个tile的列数。
 * @param warps_per_block 每个block的warp数量。
 * 
 * @return 每层的 [Row, Col, Val] 三元组列表和每层的采样时间信息
 */
std::pair<std::vector<std::vector<torch::Tensor>>, std::vector<SamplingTimingResult>> bcsr_sample_all_layers_wrapper(
    const torch::Tensor& window_offset,
    const torch::Tensor& original_col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& initial_seeds,
    std::vector<int> fanouts,
    long long num_global_nodes,
    int tile_rows,
    int tile_cols,
    int warps_per_block)
{
    // 使用 vector 存储每一层的时间
    std::vector<SamplingTimingResult> all_layer_timings;
    std::vector<std::vector<torch::Tensor>> all_layers_data;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // 1. 准备第一层的 active windows 和 masks
    auto win_mask_pair = prepare_windows_and_masks(initial_seeds, tile_rows, stream);
    torch::Tensor curr_active_windows = win_mask_pair.first;
    torch::Tensor curr_masks = win_mask_pair.second;
    
    // 2. 循环采样
    for (int fanout : fanouts) {
        // 调用核心采样
        auto res = sample_one_layer_internal(
            window_offset, original_col_indices, values_condensed,
            curr_active_windows, curr_masks,
            fanout,
            tile_rows, tile_cols, warps_per_block, stream
        );
        
        auto tensors = res.first;
        auto timing = res.second; // 当前层的计时
        
        // 保存每一层的详细时间
        all_layer_timings.push_back(timing);
        
        // 将当前层使用的 active_windows 和 masks 都返回给 Python
        tensors.push_back(curr_active_windows); 
        tensors.push_back(curr_masks); // 现在 tensors 长度变为 5: [win, col, val, act_win, act_mask]
        all_layers_data.push_back(tensors);

        // 3. 为下一层准备 (使用当前层采样出的列 tensors[1])
        if (&fanout != &fanouts.back()) {
             auto next_pair = prepare_windows_and_masks(tensors[1], tile_rows, stream);
             curr_active_windows = next_pair.first;
             curr_masks = next_pair.second;
        }
    }
    
    return {all_layers_data, all_layer_timings};
}

