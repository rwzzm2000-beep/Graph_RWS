# Graph_RWS/helper/logger.py

import json


# -------------------- 表格日志记录器 --------------------
class TableLogger:
    def __init__(self):
        # 定义列宽配置，方便调整
        self.header_fmt = (
            "{:^3} | {:^7} | {:^10} | {:^8} | {:^8} | "  # Ep, Time, Loss, TrAcc, VaAcc
            "{:^8} | {:^11} | {:^8} | {:^8} | {:^9} | {:^9}" # Data, Sample, Fwd, Bwd, Step, SpMM
        )
        self.row_fmt = (
            "{:03d} | {:7.3f} | {:10.4f} | {:8} | {:8} | "
            "{:8.3f} | {:11.3f} | {:8.3f} | {:8.3f} | {:9.3f} | {:9.3f}"
        )
        self.columns = [
            'Ep', 'Time(s)', 'Loss', 'TrAcc(%)', 'VaAcc(%)', 
            'Data(ms)', 'Sample(ms)', 'Fwd(ms)', 'Bwd(ms)', 'Step(ms)', 'SpMM(ms)'
        ]
        self.epoch_times = []


    def print_header(self):
        header_str = self.header_fmt.format(*self.columns)
        print("\n" + header_str)
        print("-" * len(header_str))


    def log_epoch(self, epoch, dt, loss, train_acc, val_acc, stats):
        """
        优雅地打印一行训练日志
        """
        self.epoch_times.append(dt)

        # 处理 None 值显示
        val_str = f"{val_acc*100:5.2f}" if val_acc is not None else "  -  "
        train_str = f"{train_acc*100:5.2f}"
        
        # 打印
        print(self.row_fmt.format(
            epoch, dt, loss, train_str, val_str,
            stats['Data'], stats['Sample'], stats['Fwd'], 
            stats['Bwd'], stats['Step'], stats['SpMM']
        ))

    def print_avg_epoch_time(self):
        """
        打印表格底部的统计信息（平均时间）
        """
        if not self.epoch_times:
            return 0.0
        
        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        
        print(f"\nAverage Epoch Time: {avg_time:.3f} s")
        

def log_config_info(config):
    enable_save_model = config.get('save_model', False)
    enable_save_result = config.get('save_result', True)
    enable_sampling = config.get('sampling', False)
    sampling_batch_size = config.get('batch_size', 1024)

    print("Configuration:")
    for key, value in config.items():
        if key == 'dataset' and isinstance(value, dict):
            # 展开 dataset 字典，不显示一整行 JSON
            print(f"  dataset:")
            for ds_k, ds_v in value.items():
                print(f"    {ds_k}: {ds_v}")
        elif isinstance(value, dict):
            # 其他字典还是为了简洁显示为一行，或者你也想展开
            print(f"  {key}: {json.dumps(value, default=str)}")
        else:
            print(f"  {key}: {value}")

    print("\nRun Settings:")
    print(f"  Save Model:  {enable_save_model}")
    print(f"  Save Result: {enable_save_result}")
    if enable_sampling:
        print(f"  Sampling:    ENABLED (Batch Size: {sampling_batch_size})")
    else:
        print(f"  Sampling:    DISABLED (Full Graph)")


def log_dataset_info(bundle, config):
    ds_cfg = config.get('dataset', {})
    ds_name = ds_cfg.get('class') or ds_cfg.get('name') or ds_cfg.get('path') or "Unknown"
    # 如果是路径，取文件名
    if '/' in ds_name: 
        ds_name = os.path.basename(ds_name).split('.')[0]

    print(f"\n" + "-" * 50)
    print(f"\nDataset Name: {ds_name}")
    print(f"Status: Loaded Successfully.")
    print(f"  Nodes: {bundle.bcsr_full.num_nodes}, Edges: {bundle.bcsr_full.num_edges}")
    print(f"  Features Dim: {bundle.features.shape[1]}, Classes: {bundle.num_classes}")
    print(f"  Split: Train({len(bundle.train_idx)}) / Val({len(bundle.val_idx)}) / Test({len(bundle.test_idx)})")

    return ds_name