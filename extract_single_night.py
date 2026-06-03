# -*- coding: utf-8 -*-
"""
extract_single_night.py — 单夜特征提取子进程
=============================================
被 run_pipeline.py 以 subprocess 调用，独立运行一个夜晚的特征提取。
避免 importlib 内联加载时的阻塞问题。

用法:
  python extract_single_night.py --mff I:/101Night/Nathalie-40_...mff --night 40
"""

import sys
import gc
import time
import argparse
from pathlib import Path

# ── 路径 ──
PROJECT_DIR = Path(__file__).resolve().parent

# 轻量特征组 (无需多通道、SSSM、tsfresh)
LIGHT_GROUPS = ('basic', 'time_domain', 'frequency', 'entropy', 'complexity')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None, help='输出目录 (默认: 当前目录)')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night
    mff_path = args.mff

    print(f"[{night}] 开始特征提取: {mff_path}")
    t0 = time.time()

    try:
        # 导入 SleepEEGFeatureExtractor (importlib, 数字文件名)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "analy", str(PROJECT_DIR / "101night_analy.py"))
        analy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analy)
        SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor

        print(f"[{night}] 模块加载完成 ({time.time()-t0:.0f}s)")

        # 初始化 — 仅单通道，不加载多通道
        ext = SleepEEGFeatureExtractor(
            mff_path, eeg_channel='E21',
            load_all_channels=False,
        )
        print(f"[{night}] 初始化: {ext.n_channels}ch, {ext.sfreq:.0f}Hz, "
              f"{ext.n_times/ext.sfreq/3600:.1f}h ({time.time()-t0:.0f}s)")

        # 运行轻量特征组
        ext.run_all(
            epoch_sec=30,
            groups=LIGHT_GROUPS,
            skip_yasa=True,
            skip_sssm=True,
        )
        print(f"[{night}] run_all 完成 ({time.time()-t0:.0f}s)")

        # 保存 CSV
        df = ext.to_dataframe(epoch_sec=30)
        csv_path = out_dir / f"features_night{night}_pipeline.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{night}] CSV 保存: {csv_path} ({len(df)} epochs, {len(df.columns)} cols)")

        # 保存 pickle
        pkl_path = out_dir / f"features_night{night}_pipeline.pkl"
        ext.save_features(str(pkl_path))
        print(f"[{night}] PICKLE 保存: {pkl_path}")

        elapsed = time.time() - t0
        print(f"[{night}] ✓ 完成 ({elapsed:.0f}s)")
        return 0

    except Exception as e:
        print(f"[{night}] ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
