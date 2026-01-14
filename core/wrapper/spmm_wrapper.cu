// PyTorch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// standard library
#include <vector>
#include <iostream>

// CUDA library
#include <cuda_runtime.h>

// Custom headers
#include "../helper/macros.h"
#include "../helper/timer.h"
#include "../helper/config.h"
#include "../include/spmm_tc.cuh"

using namespace zhao;


/**
 * @brief PyTorch 绑定的 SpMM (TC) 函数
 * 
 * @param window_offset BCSR 格式的 window_offset 张量
 * @param original_col_indices BCSR 格式的 original_col_indices 张量
 * @param values_condensed BCSR 格式的 values_condensed 张量
 * @param input_B 输入的密集矩阵 B
 * @param warps_per_block 每个 Block 的 Warp 数量，用于控制并行度
 * @return 一个pair: <torch::Tensor, double>
 * @return 第一个元素是计算结果密集矩阵 C
 * @return 第二个元素是 SpMM 时间
 */
std::pair<torch::Tensor, double> spmm_tc_wrapper(
    const torch::Tensor& window_offset,
    const torch::Tensor& original_col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& input_B,
    int warps_per_block)
{
    CUDATimer timer;
    timer.start();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 输入检查
    // 适配 UVA 模式：图结构允许在 CPU 或 GPU
    // 移除 CHECK_INPUT (因为它包含 CHECK_CUDA)，只检查内存连续性
    TORCH_CHECK(window_offset.is_contiguous(), "window_offset must be contiguous");
    TORCH_CHECK(original_col_indices.is_contiguous(), "original_col_indices must be contiguous");
    TORCH_CHECK(values_condensed.is_contiguous(), "values_condensed must be contiguous");
    CHECK_INPUT(input_B);

    // 获取维度信息
    // window_offset 是 int64 类型 (long long)，大小为 num_windows + 1
    const long long num_windows = window_offset.size(0) - 1;
    
    // 使用 zhao 命名空间中的常量 (定义在 config.h)
    // M = num_windows * tile_M
    const long long M = num_windows * zhao::tile_M; 
    const long long N = input_B.size(1);

    // 3. 准备输出张量 (M x N)
    auto output_C = torch::zeros({M, N}, input_B.options());

    // 4. 配置并启动 CUDA Kernel
    // warps_per_block 必须与 spmm_tc.cuh 中的设计匹配 (通常为 8)
    dim3 grid(num_windows);
    dim3 block(32, warps_per_block);

    // 5. 获取数据指针
    // spmm_tc.cuh 中的 Kernel 签名接受 const long long*，二者在 64 位系统上兼容
    const long long* d_window_offset = GET_CONST_PTR(long long, window_offset);
    const int* d_original_col_indices = GET_CONST_PTR(int, original_col_indices);
    const float* d_values_condensed = GET_CONST_PTR(float, values_condensed);
    const float* d_input_B = GET_CONST_PTR(float, input_B);
    float* d_output_C = GET_PTR(float, output_C);

    // 6. 调用核函数
    zhao::tcspmm_base_kernel<<<grid, block, 0, stream>>>(
        N,
        d_window_offset,
        d_original_col_indices,
        d_values_condensed,
        d_input_B,
        d_output_C
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));

    timer.stop();

    return {output_C, timer.elapsed()};
}


/**
 * @brief Transpose SpMM Wrapper
 * 
 * @param window_offset BCSR 格式的 window_offset 张量
 * @param original_col_indices BCSR 格式的 original_col_indices 张量
 * @param values_condensed BCSR 格式的 values_condensed 张量
 * @param grad_out 输入梯度 [num_rows, K]
 * @param num_cols 输入特征维度 K
 *
 * @return 输出梯度 [num_cols, K]
 */
torch::Tensor spmm_transpose_wrapper(
    const torch::Tensor& window_offset,
    const torch::Tensor& original_col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& grad_out,
    int num_cols)
{
    // 适配 UVA 模式：图结构允许在 CPU 或 GPU
    TORCH_CHECK(window_offset.is_contiguous(), "window_offset must be contiguous");
    TORCH_CHECK(original_col_indices.is_contiguous(), "original_col_indices must be contiguous");
    TORCH_CHECK(values_condensed.is_contiguous(), "values_condensed must be contiguous");
    CHECK_INPUT(grad_out);

    long long num_windows = window_offset.size(0) - 1;
    int feat_dim = grad_out.size(1);
    
    // 初始化输出梯度 grad_in，必须全 0 (因为用 atomicAdd)
    auto opts = grad_out.options();
    auto grad_in = torch::zeros({num_cols, feat_dim}, opts);

    // 配置 Kernel
    // Block: 按 Feature Dim 并行 (假设 feat_dim <= 1024)
    int threads = 256; 
    if (feat_dim <= 1024) threads = feat_dim;
    
    dim3 block(threads);
    dim3 grid(num_windows); // 每个 Window 一个 Block

    // 获取 Stream (配合流水线)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    zhao::spmm_bcsr_transpose_atomic_kernel<16, 8><<<grid, block, 0, stream>>>(
        num_windows,
        feat_dim,
        GET_CONST_PTR(long long, window_offset),
        GET_CONST_PTR(int, original_col_indices),
        GET_CONST_PTR(float, values_condensed),
        GET_CONST_PTR(float, grad_out),
        GET_PTR(float, grad_in)
    );
    
    //CHECK_CUDA(cudaGetLastError());
    return grad_in;
}
