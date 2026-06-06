# -*- coding: utf-8 -*-
"""
step2_yasa_staging.py — Step 2: YASA 睡眠分期 (独立进程)
========================================================
通过 MNE 加载 EOG (E67)，与 self.data (E21) 同源同长，
送 YASA SleepStaging 做 EEG+EOG 五期分类 (Wake/N1/N2/N3/REM)。

进程退出后 OS 自动回收全部内存。

用法:
  python step2_yasa_staging.py --mff I:/.../Nathalie-40_...mff --night 40
  python step2_yasa_staging.py --mff ... --night 40 --eog-channel E219
"""

import sys
import gc
import time
import pickle
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Step 2: YASA 睡眠分期')
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None, help='输出目录 (默认: 当前目录)')
    parser.add_argument('--eeg-channel', default='E21')
    parser.add_argument('--eog-channel', default='E67')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night

    print(f"\n{'#'*60}")
    print(f"# Step 2 — YASA 睡眠分期 (Night {night})")
    print(f"# EEG: {args.eeg_channel}, EOG: {args.eog_channel}")
    print(f"{'#'*60}\n")
    t0 = time.time()

    try:
        import feature_101night_analy as analy
        SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor

        ext = SleepEEGFeatureExtractor(
            args.mff, eeg_channel=args.eeg_channel,
            load_all_channels=False)
        print(f"[N{night}] 初始化 ({time.time()-t0:.0f}s)")

        ext.run_step2_yasa_with_eog(
            eog_channel=args.eog_channel)

        pkl_path = out_dir / f"features_night{night}_step2.pkl"
        ext.save_features(str(pkl_path))
        print(f"[N{night}] Step 2 保存: {pkl_path}")

        elapsed = time.time() - t0
        print(f"[N{night}] ✓ Step 2 完成 ({elapsed/60:.1f} 分钟)")
        return 0

    except Exception as e:
        print(f"[N{night}] ✗ Step 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == '__main__':
    sys.exit(main())
