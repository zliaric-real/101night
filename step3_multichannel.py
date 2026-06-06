# -*- coding: utf-8 -*-
"""
step3_multichannel.py — Step 3: 多通道 ROI 特征提取 (独立进程)
==============================================================
半球分区 (≤5通道/侧) 加载 ~10 个代表通道，提取：
功能连接、微状态、全局场功率、空间复杂度、图论指标。
完成后 OS 进程退出自动回收全部内存。

用法:
  python step3_multichannel.py --mff I:/.../Nathalie-40_...mff --night 40
  python step3_multichannel.py --mff ... --night 40 --max-per-hemi 3
"""

import sys
import gc
import time
import pickle
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Step 3: 多通道 ROI 特征')
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None, help='输出目录 (默认: 当前目录)')
    parser.add_argument('--eeg-channel', default='E21')
    parser.add_argument('--epoch-sec', type=float, default=30)
    parser.add_argument('--max-per-hemi', type=int, default=5,
                        help='每半球最多通道数 (默认: 5)')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night

    print(f"\n{'#'*60}")
    print(f"# Step 3 — 多通道 ROI 特征 (Night {night})")
    print(f"# 半球 ≤{args.max_per_hemi}通道/侧, Epoch: {args.epoch_sec}s")
    print(f"{'#'*60}\n")
    t0 = time.time()

    try:
        import feature_101night_analy as analy
        SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor

        ext = SleepEEGFeatureExtractor(
            args.mff, eeg_channel=args.eeg_channel,
            load_all_channels=True)  # 允许惰性多通道加载
        print(f"[N{night}] 初始化: {ext.n_channels}通道, {ext.sfreq}Hz "
              f"({time.time()-t0:.0f}s)")

        ext.run_step3_multichannel_roi(
            epoch_sec=args.epoch_sec,
            max_per_hemi=args.max_per_hemi,
            groups=('connectivity', 'microstates', 'spatial'))

        pkl_path = out_dir / f"features_night{night}_step3.pkl"
        ext.save_features(str(pkl_path))
        print(f"[N{night}] Step 3 保存: {pkl_path}")

        elapsed = time.time() - t0
        print(f"[N{night}] ✓ Step 3 完成 ({elapsed/60:.1f} 分钟)")
        return 0

    except Exception as e:
        print(f"[N{night}] ✗ Step 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == '__main__':
    sys.exit(main())
