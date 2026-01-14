# Graph_RWS/timer.py
import time

# ==========================================
# 1. SpMM Kernel 计时 (Global Helper)
# ==========================================
_SPMM_TIMINGS = []

def record_spmm_time(t):
    """由 layer.py 调用，记录一次 SpMM 执行时间"""
    _SPMM_TIMINGS.append(t)

def reset_spmm_timings():
    global _SPMM_TIMINGS
    _SPMM_TIMINGS = []

def get_avg_spmm_time():
    if not _SPMM_TIMINGS: return 0.0
    return sum(_SPMM_TIMINGS) / len(_SPMM_TIMINGS)

# ==========================================
# 2. 全局阶段计时器 (Stage Timer)
# ==========================================
class StageTimer:
    """用于统计 Stage 1, Stage 2 等大阶段的耗时"""
    def __init__(self):
        self.start_time = None
        self.stage_name = None

    def start_stage(self, name):
        if self.stage_name is not None:
            self.end_stage()
        self.stage_name = name
        self.start_time = time.time()
        print(f"\n" + "=" * 80)
        print(f" {name}")
        print("=" * 80)

    def end_stage(self):
        if self.stage_name and self.start_time:
            cost = time.time() - self.start_time
            print(f"\nStage Time Cost: {cost:.4f} s")
            self.stage_name = None
            self.start_time = None

# ==========================================
# 3. 训练周期计时器 (Epoch Timer)
# ==========================================
class EpochTimer:
    """用于统计一个 Epoch 内各个环节的平均耗时 (ms)"""
    def __init__(self):
        self.keys = ['Data', 'Sample', 'Fwd', 'Bwd', 'Step']
        self.reset()

    def reset(self):
        self.metrics = {k: 0.0 for k in self.keys}
        self.counts = 0
        self.t_mark = 0.0
        reset_spmm_timings()

    def start_batch(self):
        self.t_mark = time.time()

    def record(self, key):
        """记录从上一次 record/start 到现在的耗时，并累加到 key 中"""
        now = time.time()
        self.metrics[key] += (now - self.t_mark)
        self.t_mark = now

    def end_batch(self):
        self.counts += 1

    def get_epoch_stats(self):
        """返回平均耗时 (ms) 字典"""
        # 无论 counts 是否为 0，都获取 SpMM 时间
        # 因为 SpMM 计时是独立于 batch 计数的全局统计
        spmm_time = get_avg_spmm_time()

        if self.counts == 0:
            # Full Graph 模式下没有调用 end_batch，counts 为 0
            # 返回全 0 的基础指标，并补上 SpMM 时间
            stats = {k: 0.0 for k in self.metrics}
            stats['SpMM'] = spmm_time
            return stats
        
        stats = {k: (v / self.counts) * 1000 for k, v in self.metrics.items()}
        stats['SpMM'] = spmm_time # 单独获取 SpMM
        return stats