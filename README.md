================================================================================
         Graph_RWS: High-Performance GCN with BCSR & Tensor Cores
================================================================================

[项目简介]
Graph_RWS 是一个基于 NVIDIA GPU Tensor Cores 加速的高性能图神经网络训练框架。
该项目采用 BCSR (Blocked Compressed Sparse Row) 格式存储稀疏图，通过自定义
的 CUDA Kernel 实现了高效的稀疏矩阵乘法 (SpMM) 和图采样操作。

核心特性：
  1. 极致性能：利用 WMMA (Warp Matrix Multiply Accumulate) 指令在 Tensor Core
     上执行 SpMM，显著优于传统 CUDA 核心实现。
  2. 显存优化：基于 BCSR 格式的紧凑存储，并内置自动对齐 (Padding) 机制。
  3. 灵活易用：兼容 DGL、OGB 数据集及 Matrix Market (.mtx) 文件格式。

================================================================================
# 1. 环境配置要求 (Prerequisites)
================================================================================

建议在 Linux 环境下运行，并确保拥有 Compute Capability >= 7.0 的 NVIDIA GPU
(如 Volta, Turing, Ampere 架构，推荐 RTX 3090/4090)。

核心依赖：
  * Python >= 3.8
  * CUDA Toolkit >= 11.0 (需要 nvcc 编译器)
  * PyTorch >= 1.10 (CUDA 版)

Python 库安装：
  pip install torch numpy scipy pyyaml
  pip install dgl      # (可选：如果使用 DGL 数据集)
  pip install ogb      # (可选：如果使用 OGB 数据集)

================================================================================
# 2. 项目文件结构 (Project Structure)
================================================================================

Graph_RWS/
|-- configs/                        # [配置] 存放训练参数配置文件 (.yaml)
|   |-- cora.yaml                   # Cora 数据集标准配置
|   |-- ogbn_products.yaml          # OGBN-Products 大图配置示例
|
|-- core/                           # [核心] C++/CUDA 算子扩展目录
|   |-- include/                    # CUDA 头文件 (.cuh)
|   |   |-- spmm_tc.cuh             # [核心1] Tensor Core SpMM 算子实现
|   |   |-- sampling.cuh            # [核心2] GPU 并行采样算子实现
|   |   |-- zhao.h                  # 统一头文件接口
|   |   `-- ...
|   |-- wrapper/                    # PyTorch C++ 绑定 (.cpp/.cu)
|   |   |-- spmm_wrapper.cu         # SpMM 的 Python 接口绑定
|   |   |-- sampling_wrapper.cu     # Sampling 的 Python 接口绑定
|   |   `-- bind.cpp                # Pybind11 模块注册入口
|   |-- setup.py                    # 编译脚本 (setuptools)
|
|-- data/                           # [数据] 数据处理与加载模块
|   |-- bcsr.py                     # BCSRGraph 类定义 (核心数据结构)
|   |                               # 负责 COO->CSR->BCSR 转换及转置缓存
|   |-- data_loader.py              # 数据适配器 (DGL/OGB/MTX -> BCSR)
|                                   # 负责 Features 的 Padding 对齐
|
|-- layers/                         # [模型层] GNN 组件
|   |-- layer.py                    # BCSRGraphConv 层定义
|                                   # 包含支持 Autograd 的 Function 封装
|
|-- models/                         # [模型] 完整网络定义
|   |-- model.py                    # GCN_BCSR 模型主程序
|                                   # 支持多层堆叠、残差连接与 Dropout
|
|-- sampler/                        # [采样] 上层采样接口
|   |-- sampler.py                  # 封装底层的采样算子
|
|-- train.py                        # [入口] 训练主脚本
`-- requirements.txt                # 依赖列表

================================================================================
# 3. 核心算子说明 (Core Kernels)
================================================================================

本框架的性能核心在于 `core/include/` 下的两个关键 CUDA 核函数：

--------------------------------------------------------------------------------
[Core 1] tcspmm_base_kernel (in spmm_tc.cuh)
--------------------------------------------------------------------------------
描述：
    基于 Tensor Core 的稀疏-稠密矩阵乘法 (SpMM)，计算 C = A * B。
    其中 A 为 BCSR 格式的稀疏邻接矩阵，B 为稠密特征矩阵。

实现细节：
    * Block 级并行：每个 CUDA Block 负责处理图的一个 Window (行块)。
    * Warp 级计算：每个 Warp 利用 `wmma.sync` 指令完成 16x16x16 或类似的
      矩阵块乘法累加。
    * 访存优化：结合 Cooperative Groups 进行高效的数据加载。
    * 边界安全：内置边界检查逻辑，处理 Padding 产生的无效列索引 (-1)，
      防止显存越界访问 (Illegal Memory Access)。

--------------------------------------------------------------------------------
[Core 2] sampling_kernel (in sampling.cuh)
--------------------------------------------------------------------------------
描述：
    在 GPU 上执行高效的邻居采样算法 (类似 GraphSAGE 采样)。

实现细节：
    * 并行策略：为每个目标节点分配独立的线程或 Warp 进行邻居检索。
    * 显存复用：直接在 GPU 显存中的图结构 (BCSR/CSR) 上游走，避免了
      CPU-GPU 之间频繁的数据拷贝。
    * 随机性：利用 cuRAND 库在设备端生成随机数以选择邻居。

================================================================================
4. 编译与运行 (Build & Run)
================================================================================

由于包含自定义 CUDA 扩展，首次运行前必须进行编译。

[Step 1] 编译 CUDA 扩展
    cd Graph_RWS/core
    # 清理旧编译文件 
    rm -rf build
    # 设置并行编译的作业数（根据你的 CPU 核数调整，例如 8 或 16）[可选]
    export MAX_JOBS=8
    python setup.py build_ext --inplace
    cd ../..

    # 如果修改了 core/ 下的代码，请先运行 `rm -rf build` 清理旧文件。

[Step 2] 运行训练
    CUDA_VISIBLE_DEVICES=2 python -m Graph_RWS.train --config Graph_RWS/configs/cora.yaml
    CUDA_VISIBLE_DEVICES=2 python -m Graph_RWS.train --config Graph_RWS/configs/ogbn_arxiv.yaml
    CUDA_VISIBLE_DEVICES=1 python -m Graph_RWS.train --config Graph_RWS/configs/reddit.yaml

[Step 3] 自定义配置运行
    python -m Graph_RWS.train --config ./my_config.yaml --save-model

[Step 4] 运行保存的模型
python -m Graph_RWS.test_save_model \
  --config ./Graph_RWS/configs/ogbn_products.yaml \
  --checkpoint ./Graph_RWS/save_models/ogbn-products_Full_20260106_171242/model_epoch_5.pt

python -m Graph_RWS.test_save_model \
  --config ./Graph_RWS/configs/reddit.yaml \
  --checkpoint ./Graph_RWS/save_models/RedditDataset_Sample_20251220_110100/model_epoch_100.pt

