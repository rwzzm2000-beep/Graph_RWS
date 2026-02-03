下面是**在不修改任何原始内容含义与文字的前提下**，将你的 `README.md` 调整为 **GitHub 能正常渲染、层次清晰、观感更专业美观** 的 Markdown 版本。
你可以 **直接整体复制** 覆盖原来的 `README.md`。

---

````markdown
# Graph_RWS: High-Performance GCN with BCSR & Tensor Cores

---

## 项目简介

**Graph_RWS** 是一个面向 **BCSR 稀疏图** 的 GNN 训练框架：

- **Python 侧**
  - 提供 `BCSRGraph`
  - 统一的数据加载接口 `load_dataset`
  - 将 **DGL / OGB** 图数据转换为 **BCSR**
  - `train.py` 组织全图训练与采样训练流程

- **核心计算**
  - 自定义 **C++ / CUDA 扩展**
  - Tensor Core SpMM（`spmm_tc / spmm_transpose`）
  - 多层采样（`sample_all_layers`）
  - 子图重索引（`reindex_bcsr`）

- **模型侧**
  - `BCSRGraphConv`
  - `GCN_BCSR`
  - 支持 **Residual / LayerNorm / Dropout**
  - 支持 **UVA / CPU 特征策略**

---

## 核心特性

1. **Tensor Core SpMM**  
   `core/include/spmm_tc.cuh` 基于 **WMMA / TF32** 的 BCSR SpMM，  
   `layers/BCSRGraphConv` 直接调用以加速前向与反向。

2. **CUDA 采样与重索引**  
   - `sampling.cuh` + `sampler/BCSRSampler`：多层 fanout 采样  
   - `reindex.cuh`：本地重索引与剪枝，输出可直接用于下一层计算

3. **训练与数据链路**  
   - `data_loader.py`：统一 DGL / OGB 数据集适配  
   - `engine.py`：支持全图训练、节点采样、DPS / BAPS 等 mini-batch 策略  
   - 配套多种 DataLoader

---

## 1. 环境配置要求（Prerequisites）

建议在 **Linux** 环境下运行，并确保拥有 **Compute Capability ≥ 7.0** 的 NVIDIA GPU  
（如 **Volta / Turing / Ampere** 架构，推荐 **RTX 3090 / 4090**）。

### 核心依赖

- Python ≥ 3.8  
- CUDA Toolkit ≥ 11.0（需要 `nvcc`）  
- PyTorch ≥ 1.10（CUDA 版）

### Python 库安装

```bash
pip install torch numpy scipy pyyaml
pip install dgl      # 可选：如果使用 DGL 数据集
pip install ogb      # 可选：如果使用 OGB 数据集
````

---

## 2. 项目文件结构（Project Structure）

```text
Graph_RWS/
├── configs/                        # [配置] 训练 / 数据集配置文件 (.yaml)
│   ├── amazon_computer.yaml
│   ├── citeseer.yaml
│   ├── cora.yaml
│   ├── flickr.yaml
│   ├── ogbn_arxiv.yaml
│   ├── ogbn_papers100M.yaml
│   ├── ogbn_products.yaml
│   ├── ppi.yaml
│   ├── pubmed.yaml
│   ├── reddit.yaml
│   └── yelp.yaml
│
├── core/                           # [核心] C++ / CUDA 算子扩展
│   ├── README.md
│   ├── setup.py
│   ├── __init__.py
│   │
│   ├── helper/
│   │   ├── config.h
│   │   ├── macros.h
│   │   ├── sampling_helper.h
│   │   └── timer.h
│   │
│   ├── include/
│   │   ├── spmm_tc.cuh
│   │   ├── sampling.cuh
│   │   └── reindex.cuh
│   │
│   └── wrapper/
│       ├── bind.cpp
│       ├── spmm_wrapper.cu
│       ├── sampling_wrapper.cu
│       ├── reindex_wrapper.cu
│       └── convert_wrapper.cpp
│
├── data/                           # [数据] 数据处理与加载
│   ├── bcsr.py
│   ├── data_loader.py
│   ├── __init__.py
│   └── .gitkeep
│
├── layers/                         # [层] GNN 基础层
│   ├── layer.py
│   └── __init__.py
│
├── models/                         # [模型] 网络结构定义
│   ├── model.py
│   └── __init__.py
│
├── sampler/                        # [采样] 采样接口
│   ├── sampler.py
│   └── __init__.py
│
├── helper/                         # [工具] 日志 / 计时 / 通用工具
│   ├── engine.py
│   ├── logger.py
│   ├── timer.py
│   ├── utils.py
│   └── __init__.py
│
├── train.py                        # [入口] 训练脚本
├── test_save_model.py              # [测试] 模型保存 / 加载测试
├── requirements.txt
├── README.md
├── .gitignore
└── .vscode/
    └── extensions.json
```

---

## 3. 编译与运行（Build & Run）

由于包含自定义 CUDA 扩展，**首次运行前必须编译**。

### Step 1：编译 CUDA 扩展

```bash
cd Graph_RWS/core

# 清理旧编译文件
rm -rf build

# 设置并行编译任务数（可选，例如 8 或 16）
export MAX_JOBS=8

python setup.py build_ext --inplace
cd ../..
```

> 如果修改了 `core/` 下的代码，请先运行 `rm -rf build`。

---

### Step 2：运行训练

```bash
CUDA_VISIBLE_DEVICES=2 python -m Graph_RWS.train --config Graph_RWS/configs/cora.yaml
CUDA_VISIBLE_DEVICES=2 python -m Graph_RWS.train --config Graph_RWS/configs/ogbn_arxiv.yaml
CUDA_VISIBLE_DEVICES=1 python -m Graph_RWS.train --config Graph_RWS/configs/reddit.yaml
CUDA_VISIBLE_DEVICES=1 python -m Graph_RWS.train --config Graph_RWS/configs/ogbn_papers100M.yaml
```

---

### Step 3：自定义配置运行

```bash
python -m Graph_RWS.train --config ./my_config.yaml --save-model
```

---

### Step 4：运行保存的模型

```bash
python -m Graph_RWS.test_save_model \
  --config ./Graph_RWS/configs/ogbn_products.yaml \
  --checkpoint ./Graph_RWS/save_models/ogbn-products_Full_20260106_171242/model_epoch_5.pt
```

```bash
python -m Graph_RWS.test_save_model \
  --config ./Graph_RWS/configs/reddit.yaml \
  --checkpoint ./Graph_RWS/save_models/RedditDataset_Sample_20251220_110100/model_epoch_100.pt
```
