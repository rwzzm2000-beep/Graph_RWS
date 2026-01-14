#pragma once

// standard library
#include <cstdlib>
#include <iostream>

// PyTorch
#include <torch/extension.h>


namespace zhao {


// ================= 标准化指针获取宏 =================
// 统一使用这个宏来获取 Tensor 的 Raw Pointer
#define GET_PTR(TYPE, TENSOR) ((TYPE*)TENSOR.data_ptr())
#define GET_CONST_PTR(TYPE, TENSOR) ((const TYPE*)TENSOR.data_ptr())


// Error checking helpers
#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t error = (call);                                                                                    \
        if (error != cudaSuccess) {                                                                                    \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error)          \
                      << std::endl;                                                                                    \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)


// PyTorch Tensor Checks
#define CHECK_CUDA_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA_TENSOR(x); CHECK_CONTIGUOUS(x)


// =========================================================
// 采样 Kernel 自动分发宏
// =========================================================

// 定义具体的 Sampling Kernel 调用
// WP: Warps Per Block, CAP: Capacity (Template), ACT: Actual (Runtime)
#define CALL_SAMPLING_KERNEL(WP, CAP, ACT, STREAM, ...) \
    zhao::warp_centric_find_top_k_kernel<WP, CAP, 16, 8><<<grid, block, 0, STREAM>>>( \
        __VA_ARGS__, ACT \
    )

// 定义具体的 Recompress Kernel 调用
#define CALL_RECOMPRESS_KERNEL(WP, CAP, ACT, STREAM, ...) \
    zhao::recompress_sampled_bcsr_kernel<WP, CAP, 16, 8><<<grid, block, 0, STREAM>>>( \
        __VA_ARGS__, ACT \
    )

// 采样分发器：根据 ACTUAL (fanout) 自动匹配最小够用的 CAPACITY (Template)
// 参数说明：
//   WARPS: 运行时传入的 block 大小配置
//   FANOUT: 实际采样数
//   ...: 传递给 Kernel 的其他指针参数
#define DISPATCH_SAMPLING(WARPS, FANOUT, STREAM, ...) \
    [&] { \
        if (FANOUT <= 32) { \
            CALL_SAMPLING_KERNEL(WARPS, 32, FANOUT, STREAM, __VA_ARGS__); \
        } else if (FANOUT <= 128) { \
            CALL_SAMPLING_KERNEL(WARPS, 128, FANOUT, STREAM, __VA_ARGS__); \
        } else if (FANOUT <= 256) { \
            CALL_SAMPLING_KERNEL(WARPS, 256, FANOUT, STREAM, __VA_ARGS__); \
        } else { \
            TORCH_CHECK(false, "Fanout > 256 not supported yet (Add more cases in macros.h)."); \
        } \
    }()

// 重压缩分发器
// 注意：Recompress Kernel 不需要传 ACTUAL 给 Kernel 内部（它只用 Map），
// 但我们需要根据 FANOUT 选择对应的 Template 大小（因为 Shared Memory 大小依赖于它）
#define DISPATCH_RECOMPRESS(WARPS, FANOUT, STREAM, ...) \
    [&] { \
        if (FANOUT <= 32) { \
            CALL_RECOMPRESS_KERNEL(WARPS, 32, FANOUT, STREAM, __VA_ARGS__); \
        } else if (FANOUT <= 128) { \
            CALL_RECOMPRESS_KERNEL(WARPS, 128, FANOUT, STREAM, __VA_ARGS__); \
        } else if (FANOUT <= 256) { \
            CALL_RECOMPRESS_KERNEL(WARPS, 256, FANOUT, STREAM, __VA_ARGS__); \
        } else { \
            TORCH_CHECK(false, "Fanout > 256 not supported yet."); \
        } \
    }()


}   // namespace zhao

