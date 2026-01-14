// PyTorch
#include <torch/extension.h>

// standard library
#include <vector>
#include <utility>
#include <iomanip>
#include <sstream>

// custom headers
#include "../helper/timer.h" 

namespace py = pybind11;


std::pair<std::vector<std::vector<torch::Tensor>>, std::vector<zhao::SamplingTimingResult>> bcsr_sample_all_layers_wrapper(
    const torch::Tensor& window_offset,
    const torch::Tensor& original_col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& initial_seeds,
    std::vector<int> fanouts,
    long long num_global_nodes,
    int tile_rows,
    int tile_cols,
    int warps_per_block
);


std::pair<torch::Tensor, double> spmm_tc_wrapper(
    const torch::Tensor& window_offset,
    const torch::Tensor& original_col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& input_B,
    int warps_per_block
);


torch::Tensor spmm_transpose_wrapper(
    const torch::Tensor& window_offset,
    const torch::Tensor& original_col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& grad_out,
    int num_cols
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> csr_to_bcsr_cpu(
    long long num_rows,
    const torch::Tensor& row_ptr,
    const torch::Tensor& col_idx,
    const torch::Tensor& values,
    long long tile_rows,
    long long tile_cols
);


torch::Tensor reindex_bcsr_cuda(
    const torch::Tensor& col_indices,
    const torch::Tensor& values_condensed,
    const torch::Tensor& active_windows,
    int tile_rows,
    int tile_cols
);


// 统一的模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 1. 注册 SamplingTimingResult 类
    // 这样 Python 端就能得到一个拥有属性的对象，而不仅仅是字典
    py::class_<zhao::SamplingTimingResult>(m, "SamplingTimingResult")
        .def(py::init<>()) // 默认构造函数
        .def_readonly("phase1_sampling_time", &zhao::SamplingTimingResult::phase1_sampling_time)
        .def_readonly("phase2_mapping_time", &zhao::SamplingTimingResult::phase2_mapping_time)
        .def_readonly("phase3_recompression_time", &zhao::SamplingTimingResult::phase3_recompression_time)
        .def_readonly("total_time", &zhao::SamplingTimingResult::total_time)
        .def("to_dict", &zhao::SamplingTimingResult::to_dict)
        .def("__repr__", [](const zhao::SamplingTimingResult& t) {
            std::stringstream ss;
            ss << "<SamplingTimingResult total=" << std::fixed << std::setprecision(2) << t.total_time 
               << "ms (P1=" << t.phase1_sampling_time 
               << ", P2=" << t.phase2_mapping_time 
               << ", P3=" << t.phase3_recompression_time
               << ")>";
            return ss.str();
        });

    // 2. Multi-Layer Sampling (C++ Loop)
    m.def("sample_all_layers", &bcsr_sample_all_layers_wrapper, "BCSR Multi-Layer Sampling (C++ Loop)",
          py::arg("window_offset"),
          py::arg("original_col_indices"),
          py::arg("values_condensed"),
          py::arg("initial_seeds"),
          py::arg("fanouts"),
          py::arg("num_global_nodes"),
          py::arg("tile_rows")=16,
          py::arg("tile_cols")=8,
          py::arg("warps_per_block")=8,
          py::call_guard<py::gil_scoped_release>()
    );

    // 3. SpMM (Tensor Core)
    m.def("spmm_tc", &spmm_tc_wrapper, "BCSR SpMM (Tensor Core)",
          py::arg("window_offset"), 
          py::arg("original_col_indices"), 
          py::arg("values_condensed"),
          py::arg("input_B"),
          py::arg("warps_per_block")=8
    );

    // 4. SpMM (Transpose)
    m.def("spmm_transpose", &spmm_transpose_wrapper, "BCSR Transpose SpMM (Atomic)",
          py::arg("window_offset"),
          py::arg("original_col_indices"),
          py::arg("values_condensed"),
          py::arg("grad_out"),
          py::arg("num_cols"),
          py::call_guard<py::gil_scoped_release>() // 允许并行
    );

    // 5. Convert (csr to bcsr)
    m.def("csr_to_bcsr_cpu", &csr_to_bcsr_cpu, "Convert CSR to BCSR on CPU (OpenMP)",
      py::arg("num_rows"),
      py::arg("row_ptr"),
      py::arg("col_idx"),
      py::arg("values"),
      py::arg("tile_rows"),
      py::arg("tile_cols")
    );
    
    // 6. Reindex (CUDA)
    m.def("reindex_bcsr", &reindex_bcsr_cuda, "Fused BCSR Reindexing and Pruning",
        py::arg("col_indices"),
        py::arg("values_condensed"),
        py::arg("active_windows"),
        py::arg("tile_rows")=16,
        py::arg("tile_cols")=8
    );

}