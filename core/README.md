# Graph_RWS Core C++/CUDA Extension

该文件夹包含 Graph_RWS 的高性能核心算子实现，通过 PyTorch C++ Extension 方式调用，负责底层的并行图采样与矩阵运算。

## 📁 关键文件与目录说明

### 1. 核心算子 (core/include/)
包含 CUDA Kernel 的声明与模板实现，负责具体的 GPU 计算逻辑。

* **`sampling.cuh`**
    * **采样核心**：实现 Warp-Centric 的 Top-K 采样 (`warp_centric_find_top_k_kernel`) 和子图重压缩 (`recompress_sampled_bcsr_kernel`)。
* **`spmm_tc.cuh`**
    * **矩阵乘法**：基于 Tensor Core 优化的稀疏矩阵乘法 (SpMM) 算子。
* **`format_convert.cuh`**
    * **格式转换**：提供基于 Thrust 的通用 GPU 格式转换工具（如 `convert_coo_to_bcsr_gpu`）。

### 2. 调度与绑定 (core/wrapper/)
Python 与 CUDA 之间的胶水层，负责显存管理、流水线调度及 PyBind 注册。

* **`bind.cpp`**
    * **接口注册**：定义 Python 可调用的函数接口（`core_lib.sample_bcsr`, `core_lib.spmm_tc` 等）。
* **`sampling_wrapper.cu`**
    * **采样流水线**：调度完整的采样过程（Sample -> Scan -> Search -> Recompress -> Transpose），并调用 `format_convert.cuh` 处理格式。
* **`spmm_wrapper.cu`**
    * **SpMM 调度**：负责检查输入 tensor 并在 GPU 上启动 SpMM Kernel。
* **`convert_wrapper.cpp`**
    * **CPU 预处理**：提供基于 CPU OpenMP 的全图格式转换工具（`csr_to_bcsr_cpu`），用于数据加载阶段。

### 3. 基础设施 (core/helper/)
提供底层通用的宏、常量与辅助结构。

* **`macros.h`**：统一的错误检查 (`CHECK_INPUT`) 与指针获取宏 (`GET_PTR`)。
* **`config.h`**：全局编译期常量（Tile 尺寸、Warp 配置等）。
* **`timer.h`**：CUDA/CPU 高精度计时器。
* **`sampling_helper.h`**：采样所需的辅助结构体定义。

---

## 🔧 模块依赖说明

本项目采用 **显式包含 (Explicit Include)** 策略，不使用统一头文件，以降低耦合度：

* **采样模块**：`sampling_wrapper.cu` 依赖 `sampling.cuh` + `format_convert.cuh`
* **计算模块**：`spmm_wrapper.cu` 仅依赖 `spmm_tc.cuh`
* **预处理模块**：`convert_wrapper.cpp` 独立运行于 CPU 端