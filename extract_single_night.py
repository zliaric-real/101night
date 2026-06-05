# -*- coding: utf-8 -*-
"""
extract_single_night.py — 单夜特征提取子进程 (3步模式，不含源定位)
==================================================================
被 run_pipeline.py 以 subprocess 调用，独立运行一个夜晚的特征提取。

3步线性流水线：
  Step 1: 单通道特征 (E21≈Cz) — 频谱/时域/熵/复杂度/tsfresh/RQA/SSSM
  Step 2: YASA 睡眠分期 (EEG+EOG) — 加载 EOG 通道，分期后清除
  Step 3: 多通道 ROI 特征 — 半球分区 (≤5通道/侧)，提取后清除
  (Step 4 源定位已禁用 — 内存过大，暂不执行)

用法:
  python extract_single_night.py --mff I:/101Night/Nathalie-40_...mff --night 40
  python extract_single_night.py --mff ... --night 40 --skip-yasa   # 跳过 YASA
"""

import sys
import gc
import time
import argparse
from pathlib import Path

# ── 项目路径 ──
PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description='单夜特征提取 (3步流水线)')
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None,
                        help='输出目录 (默认: 当前目录)')
    parser.add_argument('--eeg-channel', default='E21',
                        help='主EEG通道 (默认: E21≈Cz)')
    parser.add_argument('--eog-channels', nargs=2, default=['E67', 'E219'],
                        help='EOG通道对 (默认: E67 E219)')
    parser.add_argument('--epoch-sec', type=float, default=30,
                        help='Epoch长度/秒 (默认: 30)')
    parser.add_argument('--max-per-hemi', type=int, default=5,
                        help='Step3每半球最多通道 (默认: 5)')
    parser.add_argument('--skip-yasa', action='store_true',
                        help='跳过 Step 2 (YASA 睡眠分期)')
    parser.add_argument('--skip-sssm', action='store_true',
                        help='跳过 SSSM 特征波检测')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night
    mff_path = args.mff

    print(f"\n{'#'*60}")
    print(f"# Night {night} — 3步特征提取")
    print(f"# 文件: {mff_path}")
    print(f"# 主通道: {args.eeg_channel}")
    print(f"# EOG: {args.eog_channels}")
    print(f"# Epoch: {args.epoch_sec}s, 每半球≤{args.max_per_hemi}通道")
    print(f"# 跳过YASA: {args.skip_yasa}, 跳过源定位: True (3步模式)")
    print(f"{'#'*60}\n")
    t0 = time.time()

    try:
        # ── 导入 SleepEEGFeatureExtractor ──
        import feature_101night_analy as analy
        SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor
        print(f"[{night}] 模块加载完成 ({time.time()-t0:.0f}s)")

        # ── 初始化 — 单通道 MNE 加载 (~60 MB) ──
        ext = SleepEEGFeatureExtractor(
            mff_path,
            eeg_channel=args.eeg_channel,
            load_all_channels=True,  # 允许多通道特征（惰性加载）
        )
        print(f"[{night}] 初始化: {ext.n_channels}通道, {ext.sfreq}Hz, "
              f"{ext.n_times/ext.sfreq/3600:.1f}h ({time.time()-t0:.0f}s)")

        # ── 运行 3 步流水线 (不含源定位) ──
        ext.run_all(
            epoch_sec=args.epoch_sec,
            groups=('basic', 'time_domain', 'frequency', 'entropy',
                    'complexity', 'connectivity', 'microstates', 'spatial',
                    'tsfresh', 'rqa', 'adv_spectral', 'autocorrelation'),
            skip_yasa=args.skip_yasa,
            skip_sssm=args.skip_sssm,
            skip_source_loc=True,          # Step 4 永久禁用
            yasa_eog_channels=tuple(args.eog_channels),
            max_per_hemi=args.max_per_hemi,
        )

        # ── 保存 CSV ──
        df = ext.to_dataframe(epoch_sec=args.epoch_sec)
        csv_path = out_dir / f"features_night{night}_pipeline.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{night}] CSV 保存: {csv_path} "
              f"({len(df)} epochs, {len(df.columns)} cols)")

        # ── 保存 Pickle ──
        pkl_path = out_dir / f"features_night{night}_pipeline.pkl"
        ext.save_features(str(pkl_path))
        print(f"[{night}] PICKLE 保存: {pkl_path}")

        elapsed = time.time() - t0
        print(f"\n[{night}] ✓ 完成 ({elapsed/60:.1f} 分钟)")
        return 0

    except Exception as e:
        print(f"[{night}] ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == '__main__':
    sys.exit(main())
