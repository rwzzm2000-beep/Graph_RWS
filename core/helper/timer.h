#pragma once

// standard headers
#include <map>
#include <algorithm>
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

// CUDA headers
#include <cuda_runtime.h>

namespace zhao {


/**
 * @brief CUDA计时器
 * 
 * 用于测量CUDA内核执行时间
 */
class CUDATimer {
public:
    CUDATimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
    }

    ~CUDATimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }

    void start() { 
        cudaEventRecord(start_event); 
    }

    void stop() {
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
    }

    double elapsed() {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, end_event);
        return static_cast<double>(milliseconds);
    }

private:
    cudaEvent_t start_event;
    cudaEvent_t end_event;
};


/**
 * @brief CPU计时器
 * 
 * 用于测量CPU代码执行时间
 */
class CPUTimer {
public:
    void start() { 
        start_time = std::chrono::high_resolution_clock::now(); 
    }

    void stop() { 
        end_time = std::chrono::high_resolution_clock::now(); 
    }

    double elapsed() const {
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
};


/**
 * @brief 采样过程的详细计时结果
 * 
 * 用于记录采样过程中各个阶段的执行时间
 */
struct SamplingTimingResult {
    double phase1_sampling_time = 0.0;
    double phase2_mapping_time = 0.0;
    double phase3_recompression_time = 0.0;
    double total_time = 0.0;
    
    // 添加便捷方法
    double get_bottleneck_time() const {
        return std::max({phase1_sampling_time, 
                        phase2_mapping_time, 
                        phase3_recompression_time});
    }
    
    std::string get_bottleneck_name() const {
        double max_time = get_bottleneck_time();
        if (max_time == phase1_sampling_time) return "Sampling";
        if (max_time == phase2_mapping_time) return "Value_Mapping";
        if (max_time == phase3_recompression_time) return "Recompression";
        return "Unknown";
    }
    
    // 转换为字典（方便Python使用）
    std::map<std::string, double> to_dict() const {
        return {
            {"phase1_sampling_time", phase1_sampling_time},
            {"phase2_mapping_time", phase2_mapping_time},
            {"phase3_recompression_time", phase3_recompression_time},
            {"total_time", total_time}
        };
    }
};


}  // namespace zhao