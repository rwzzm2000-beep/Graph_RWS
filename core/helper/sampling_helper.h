#pragma once

// standard library
#include <vector>

// CUDA library
#include <cuda_runtime.h>


namespace zhao {


// 采样配置
struct SamplingConfig {
    int sample_num = 128;               // 每个window最终采样的列数量
    int warps_per_block = 8;            // 每个Block中的Warp数量，也即批处理大小
    int large_window_threshold = 4096;  // tile数量超过此值则回退到CPU
};


// 列候选结构 (键值对)
struct ColumnCandidate {
    int column_idx;
    int frequency;
};

// 为 ColumnCandidate 创建一个显式的比较器结构体，用于 CUB 排序
struct ColumnCandidateComparator {
    __host__ __device__ __forceinline__ bool operator()(const ColumnCandidate& a, const ColumnCandidate& b) const {
        if (a.frequency != b.frequency) {
            return a.frequency > b.frequency;  // 频率降序
        }
        return a.column_idx < b.column_idx;    // 频率相同时，列索引升序
    }
};


}
