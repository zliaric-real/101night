# -*- coding: utf-8 -*-
"""
step4_source_loc.py — Step 4: 源定位 eLORETA (独立进程)
=======================================================
时间切片加载全部 260 通道，fsaverage 模板 + 3层BEM + forward，
eLORETA 逆解 → 源空间频带功率 / Desikan-Killiany ROI 功率。
完成后 OS 进程退出自动回收全部内存 (~2 GB)。

用法:
  python step4_source_loc.py --mff I:/.../Nathalie-40_...mff --night 40
  python step4_source_loc.py --mff ... --night 40 --method dSPM --slice-sec 300
"""

import sys
import gc
import time
import pickle
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Step 4: 源定位 eLORETA')
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None, help='输出目录 (默认: 当前目录)')
    parser.add_argument('--eeg-channel', default='E21')
    parser.add_argument('--epoch-sec', type=float, default=30)
    parser.add_argument('--method', default='eLORETA',
                        choices=['eLORETA', 'dSPM', 'MNE'])
    parser.add_argument('--slice-sec', type=float, default=600,
                        help='时间切片长度/秒 (默认: 600)')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night

    print(f"\n{'#'*60}")
    print(f"# Step 4 — 源定位 {args.method} (Night {night})")
    print(f"# 切片: {args.slice_sec}s, Epoch: {args.epoch_sec}s")
    print(f"{'#'*60}\n")
    t0 = time.time()

    try:
        import feature_101night_analy as analy
        SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor

        ext = SleepEEGFeatureExtractor(
            args.mff, eeg_channel=args.eeg_channel,
            load_all_channels=True)
        print(f"[N{night}] 初始化: {ext.n_channels}通道, {ext.sfreq}Hz "
              f"({time.time()-t0:.0f}s)")

        ext.run_step4_source_slices(
            epoch_sec=args.epoch_sec,
            method=args.method,
            slice_sec=args.slice_sec)

        pkl_path = out_dir / f"features_night{night}_step4.pkl"
        ext.save_features(str(pkl_path))
        print(f"[N{night}] Step 4 保存: {pkl_path}")

        elapsed = time.time() - t0
        print(f"[N{night}] ✓ Step 4 完成 ({elapsed/60:.1f} 分钟)")
        return 0

    except Exception as e:
        print(f"[N{night}] ✗ Step 4 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == '__main__':
    sys.exit(main())
