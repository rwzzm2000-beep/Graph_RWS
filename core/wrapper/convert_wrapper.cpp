// PyTorch Extension
#include <torch/extension.h>

// Standard Library
#include <vector>
#include <algorithm>
#include <omp.h>
#include <tuple>

// Custom Headers
#include "../helper/macros.h"


/**
 * @brief CSR 转 BCSR 的 CPU 并行实现 (OpenMP)
 * 
 * @param num_rows 图的总行数
 * @param row_ptr CSR 行指针
 * @param col_idx CSR 列索引
 * @param values CSR 边权重
 * @param tile_rows Tile 的行高 (通常 16)
 * @param tile_cols Tile 的列宽 (通常 8)
 *
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
 * 返回 (window_offset, original_col_indices, values_condensed)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> csr_to_bcsr_cpu(
    long long num_rows,
    const torch::Tensor& row_ptr,
    const torch::Tensor& col_idx,
    const torch::Tensor& values,
    long long tile_rows,
    long long tile_cols) 
{
    // 1. 检查输入设备 (必须在 CPU)
    TORCH_CHECK(row_ptr.device().is_cpu(), "row_ptr must be on CPU");
    
    // 2. 获取原始数据指针 (使用 C 风格强转，统一使用 long long)
    const long long* row_ptr_data = GET_PTR(long long, row_ptr);
    
    // 注意：通常 CSR 的 col_idx 是 int32 以节省显存。
    // 如果您的输入 Tensor 是 int64/long long，请将下方 int 改为 long long。
    // 基于 PyTorch 常见习惯及 bcsr.py 代码，这里默认为 int32，但逻辑计算用 long long。
    const int* col_idx_data = GET_PTR(int, col_idx); 
    const float* values_data = GET_PTR(float, values);

    long long num_windows = (num_rows + tile_rows - 1) / tile_rows;
    
    // 3. 准备输出张量选项
    auto opts_long = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    auto opts_int = torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // window_offset: [num_windows + 1]
    torch::Tensor window_offset = torch::zeros({num_windows + 1}, opts_long);
    long long* win_off_data = GET_PTR(long long, window_offset);

    // ==========================================================
    // Phase 1: 并行计算每个 Window 需要多少个 Tile (Count)
    // ==========================================================
    #pragma omp parallel for schedule(dynamic, 64)
    for (long long w = 0; w < num_windows; ++w) {
        long long r_start = w * tile_rows;
        long long r_end = std::min(r_start + tile_rows, num_rows);
        
        // 使用局部 vector 避免竞争
        std::vector<int> unique_cols;
        
        // 预估容量，减少 realloc
        long long start_edge = row_ptr_data[r_start];
        long long end_edge = row_ptr_data[r_end];
        long long est_edges = end_edge - start_edge;
        
        if (est_edges > 0) {
            unique_cols.reserve(est_edges);
            for (long long idx = start_edge; idx < end_edge; ++idx) {
                unique_cols.push_back(col_idx_data[idx]);
            }
            
            // 排序并去重
            std::sort(unique_cols.begin(), unique_cols.end());
            auto last = std::unique(unique_cols.begin(), unique_cols.end());
            long long n_unique = std::distance(unique_cols.begin(), last);
            
            // 计算需要的 tiles 数量
            long long n_tiles = (n_unique + tile_cols - 1) / tile_cols;
            win_off_data[w + 1] = n_tiles;
        } else {
            win_off_data[w + 1] = 0;
        }
    }

    // ==========================================================
    // Phase 1.5: 前缀和 (Prefix Sum) - 串行极快
    // ==========================================================
    for (long long w = 0; w < num_windows; ++w) {
        win_off_data[w + 1] += win_off_data[w];
    }
    
    long long total_tiles = win_off_data[num_windows];

    // ==========================================================
    // Allocate Final Tensors
    // ==========================================================
    // original_col_indices: [total_tiles * tile_cols]
    torch::Tensor original_col_indices = torch::full({total_tiles * tile_cols}, -1, opts_int);
    // values_condensed: [total_tiles * tile_rows * tile_cols]
    torch::Tensor values_condensed = torch::zeros({total_tiles * tile_rows * tile_cols}, opts_float);

    int* out_cols_ptr = GET_PTR(int, original_col_indices);
    float* out_vals_ptr = GET_PTR(float, values_condensed);

    // ==========================================================
    // Phase 2: 并行填充数据 (Fill)
    // ==========================================================
    #pragma omp parallel for schedule(dynamic, 64)
    for (long long w = 0; w < num_windows; ++w) {
        long long tile_start_idx = win_off_data[w];
        long long n_tiles = win_off_data[w + 1] - tile_start_idx;
        
        if (n_tiles == 0) continue;

        long long r_start = w * tile_rows;
        long long r_end = std::min(r_start + tile_rows, num_rows);
        
        // 重新收集并去重列索引 (Trade computation for memory)
        std::vector<int> unique_cols;
        long long start_edge = row_ptr_data[r_start];
        long long end_edge = row_ptr_data[r_end];
        unique_cols.reserve(end_edge - start_edge);
        
        for (long long idx = start_edge; idx < end_edge; ++idx) {
            unique_cols.push_back(col_idx_data[idx]);
        }
        std::sort(unique_cols.begin(), unique_cols.end());
        auto last = std::unique(unique_cols.begin(), unique_cols.end());
        unique_cols.erase(last, unique_cols.end());

        // 1. 填充 original_col_indices
        int* w_col_ptr = out_cols_ptr + tile_start_idx * tile_cols;
        for (size_t i = 0; i < unique_cols.size(); ++i) {
            w_col_ptr[i] = unique_cols[i];
        }

        // 2. 填充 values_condensed
        // 我们需要把 CSR 中的值填入 BCSR 的 [Tile, Row, Col] 3D 结构中
        // 展平后地址 = (GlobalTileIdx * Rows * Cols) + (IntraRow * Cols) + IntraCol
        
        // 遍历该 Window 包含的所有原始边
        for (long long r = r_start; r < r_end; ++r) {
            long long intra_row = r - r_start;
            long long r_idx_start = row_ptr_data[r];
            long long r_idx_end = row_ptr_data[r + 1];
            
            for (long long idx = r_idx_start; idx < r_idx_end; ++idx) {
                int col = col_idx_data[idx];
                float val = values_data[idx];
                
                // 二分查找该列在 unique_cols 中的位置 (Local Index)
                auto it = std::lower_bound(unique_cols.begin(), unique_cols.end(), col);
                long long pos = std::distance(unique_cols.begin(), it);
                
                // 计算其所属的 Tile 索引和 Tile 内列偏移
                long long tile_idx_local = pos / tile_cols;
                long long intra_col = pos % tile_cols;
                
                long long global_tile_idx = tile_start_idx + tile_idx_local;
                
                // 计算最终写入位置
                long long dest_idx = global_tile_idx * (tile_rows * tile_cols) 
                                     + intra_row * tile_cols 
                                     + intra_col;
                
                out_vals_ptr[dest_idx] = val;
            }
        }
    }
    
    return std::make_tuple(window_offset, original_col_indices, values_condensed);
}

