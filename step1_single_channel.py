# -*- coding: utf-8 -*-
"""
step1_single_channel.py — Step 1: 单通道特征提取 (独立进程)
==========================================================
提取 E21(≈Cz) 的频谱/时域/熵/复杂度/tsfresh/RQA/SSSM 特征。
进程退出后 OS 自动回收全部内存。

用法:
  python step1_single_channel.py --mff I:/.../Nathalie-40_...mff --night 40
  python step1_single_channel.py --mff ... --night 40 --skip-sssm
  python step1_single_channel.py --mff ... --night 40 --epoch-sec 30
"""

import sys
import gc
import time
import pickle
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Step 1: 单通道特征提取')
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None, help='输出目录 (默认: 当前目录)')
    parser.add_argument('--eeg-channel', default='E21')
    parser.add_argument('--epoch-sec', type=float, default=30)
    parser.add_argument('--skip-sssm', action='store_true', help='跳过 SSSM 特征波')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night

    print(f"\n{'#'*60}")
    print(f"# Step 1 — 单通道特征提取 (Night {night})")
    print(f"# 通道: {args.eeg_channel}, Epoch: {args.epoch_sec}s")
    print(f"{'#'*60}\n")
    t0 = time.time()

    try:
        import feature_101night_analy as analy
        SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor

        ext = SleepEEGFeatureExtractor(
            args.mff, eeg_channel=args.eeg_channel,
            load_all_channels=False)  # Step 1 只需单通道
        print(f"[N{night}] 初始化: {ext.n_channels}通道, {ext.sfreq}Hz, "
              f"{ext.n_times/ext.sfreq/3600:.1f}h ({time.time()-t0:.0f}s)")

        ext.run_step1_single_channel(
            epoch_sec=args.epoch_sec,
            groups=('basic', 'time_domain', 'frequency', 'entropy',
                    'complexity', 'tsfresh', 'rqa',
                    'adv_spectral', 'autocorrelation'),
            skip_sssm=args.skip_sssm)

        # 保存
        pkl_path = out_dir / f"features_night{night}_step1.pkl"
        ext.save_features(str(pkl_path))
        print(f"[N{night}] Step 1 保存: {pkl_path}")

        elapsed = time.time() - t0
        print(f"[N{night}] ✓ Step 1 完成 ({elapsed/60:.1f} 分钟)")
        return 0

    except Exception as e:
        print(f"[N{night}] ✗ Step 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == '__main__':
    sys.exit(main())
