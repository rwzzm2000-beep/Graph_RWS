# Expose compiled CUDA/C++ extension symbols to Python layer

try:
    # Import the compiled extension built by core/setup.py
    from . import core_lib
    
    # Expose functions with friendly names
    # 1. Multi-Layer Sampling
    sample_all_layers = getattr(core_lib, 'sample_all_layers', None)
    
    # 2. Tensor Core SpMM
    spmm_tc = getattr(core_lib, 'spmm_tc', None)

    # 3. Transpose SpMM (Atomic)
    spmm_transpose = getattr(core_lib, 'spmm_transpose', None)

    # 4. CSR to BCSR
    csr_to_bcsr_cpu = getattr(core_lib, 'csr_to_bcsr_cpu', None)

    # 5. Reindex (CUDA)
    reindex_bcsr = getattr(core_lib, 'reindex_bcsr', None)

except Exception as e:
    print("WARNING: Graph_RWS core library not compiled or failed to import:", e)
    sample_all_layers = None
    spmm_tc = None
    spmm_transpose = None
    csr_to_bcsr_cpu = None
    reindex_bcsr = None

class SamplingConfig:
    """Lightweight config container for samplers."""
    def __init__(self):
        self.sample_num = 128
        self.warps_per_block = 8
        self.large_window_threshold = 4096


__all__ = [
    'sample_all_layers',
    'spmm_tc',
    'spmm_transpose',
    'csr_to_bcsr_cpu',
    'reindex_bcsr',
    'SamplingConfig',
]
