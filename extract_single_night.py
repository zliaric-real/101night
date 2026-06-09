# -*- coding: utf-8 -*-
"""
extract_single_night.py — 单夜特征提取
=======================================
直接使用 SleepEEGFeatureExtractor (in-process, 无 subprocess)。
__init__ 已通过 MNE 一次性加载 E21+E67+半球通道并预滤波，
run_all() 线性执行 Step 1→2→3。

Step 4 源定位暂搁置，默认不调用。

用法:
  python extract_single_night.py --mff I:/101Night/Nathalie-40_...mff --night 40
  python extract_single_night.py --mff ... --night 40 --skip-yasa
  python extract_single_night.py --mff ... --night 40 --skip-sssm
"""
# -- Env: limit parallelism to avoid sklearn/joblib DLL deadlock on Windows --
import os as _os
_os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import gc
import time
import pickle
import argparse
from pathlib import Path

import feature_101night_analy as analy
SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description='单夜特征提取 (3步流水线, in-process)')
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None, help='输出目录 (默认: 当前目录)')
    parser.add_argument('--eeg-channel', default='E21')
    parser.add_argument('--eog-channel', default='E67')
    parser.add_argument('--epoch-sec', type=float, default=30)
    parser.add_argument('--max-per-hemi', type=int, default=5,
                        help='每半球代表通道数 (默认 5)')
    parser.add_argument('--skip-yasa', action='store_true',
                        help='跳过 Step 2 睡眠分期')
    parser.add_argument('--skip-sssm', action='store_true',
                        help='跳过 SSSM 特征波')
    parser.add_argument('--skip-source', action='store_true', default=True,
                        help='跳过 Step 4 源定位 (默认跳过)')
    parser.add_argument('--groups', default=None,
                        help='特征组, 逗号分隔 (默认全部)')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night

    groups = None
    if args.groups:
        groups = tuple(g.strip() for g in args.groups.split(','))

    print(f"\n{'#'*60}")
    print(f"# Night {night} — 特征提取 (3步流水线)")
    print(f"# 文件: {args.mff}")
    print(f"# EEG: {args.eeg_channel}, EOG: {args.eog_channel}")
    print(f"# 跳过 YASA: {args.skip_yasa}, 跳过 SSSM: {args.skip_sssm}")
    print(f"{'#'*60}\n")
    t_total = time.time()

    try:
        # ── 初始化 ──
        # __init__ 中一次性 MNE 加载 E21+E61+半球通道+预滤波
        t_init = time.time()
        ext = SleepEEGFeatureExtractor(
            args.mff,
            eeg_channel=args.eeg_channel,
            eog_channel=args.eog_channel,
            load_all_channels=True,
            max_per_hemi=args.max_per_hemi,
        )
        print(f"[N{night}] 初始化完成 ({time.time()-t_init:.0f}s) "
              f"— 已加载 {len(ext._pick_list)} 通道\n")

        # ── 执行 3 步流水线 ──
        run_kwargs = dict(
            epoch_sec=args.epoch_sec,
            skip_yasa=args.skip_yasa,
            skip_sssm=args.skip_sssm,
            skip_source_loc=True,
            yasa_eog_channel=args.eog_channel,
            max_per_hemi=args.max_per_hemi,
        )
        if groups:
            run_kwargs['groups'] = groups
        ext.run_all(**run_kwargs)

        # ── 保存 ──
        t_save = time.time()

        # CSV
        csv_path = out_dir / f"features_night{night}_pipeline.csv"
        df = ext.to_dataframe(epoch_sec=args.epoch_sec)
        df.to_csv(str(csv_path), index=False)
        print(f"[N{night}] CSV 保存: {csv_path} "
              f"({len(df)} epochs, {len(df.columns)} cols) "
              f"({time.time()-t_save:.0f}s)")

        # PICKLE
        pkl_path = out_dir / f"features_night{night}_pipeline.pkl"
        ext.save_features(str(pkl_path))
        print(f"[N{night}] PICKLE 保存: {pkl_path}")

        elapsed = time.time() - t_total
        print(f"\n[N{night}] ✓ 全部完成 ({elapsed/60:.1f} 分钟)")

        del ext
        gc.collect()
        return 0

    except Exception as e:
        print(f"\n[N{night}] ✗ 失败 ({time.time()-t_total:.0f}s): {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == '__main__':
    sys.exit(main())
