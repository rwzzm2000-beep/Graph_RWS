import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ABS_CORE_PATH = os.path.dirname(os.path.abspath(__file__))

INCLUDE_DIRS = [
    os.path.join(ABS_CORE_PATH, 'include'),
    os.path.join(ABS_CORE_PATH, 'helper'),
]

setup(
    name='graph_rws_core',
    ext_modules=[
        CUDAExtension(
            name='core_lib',
            sources=[
                'wrapper/bind.cpp',             # 纯 C++ 绑定代码
                'wrapper/sampling_wrapper.cu',  # 分块采样实现文件 1
                'wrapper/spmm_wrapper.cu',      # 矩阵乘法实现文件 2
                'wrapper/convert_wrapper.cpp',  # csr到bcsr
                'wrapper/reindex_wrapper.cu',   # 重索引 + 剪枝
            ],
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-fopenmp'],
                'nvcc': [
                    '-O3', 
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-D__CUDA_NO_HALF_OPERATORS__', 
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-x', 'cu' # 强制视作 CUDA 代码 (对 .cu 文件其实是默认的，但保留无妨)
                ]
            },
            extra_link_args=['-lgomp'] # 链接 OpenMP 库
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)