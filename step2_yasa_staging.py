# -*- coding: utf-8 -*-
"""
step2_yasa_staging.py — Step 2: YASA 睡眠分期
==============================================
MNE read_raw_egi → pick [E21, E67] → YASA SleepStaging
笔记本已验证路径，无需 chunked frombuffer / RawArray。

用法:
  python step2_yasa_staging.py --mff I:/.../Nathalie-40_...mff --night 40
  python step2_yasa_staging.py --mff ... --night 40 --eeg E21 --eog E219
"""
# -- Env: limit parallelism to avoid sklearn/joblib DLL deadlock on Windows --
import os as _os
_os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import gc
import time
import argparse
from pathlib import Path

import numpy as np
import mne
import yasa
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Step 2: YASA 睡眠分期')
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None, help='输出目录 (默认: 当前目录)')
    parser.add_argument('--eeg', default='E21', help='EEG 通道名')
    parser.add_argument('--eog', default='E67', help='EOG 通道名 (单侧)')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night
    eeg_ch = args.eeg
    eog_ch = args.eog

    print(f"\n{'#'*60}")
    print(f"# Step 2 — YASA 睡眠分期 (Night {night})")
    print(f"# EEG: {eeg_ch}, EOG: {eog_ch}")
    print(f"# 文件: {args.mff}")
    print(f"{'#'*60}\n")
    t0 = time.time()

    try:
        # ── 1. 加载原始数据 (单次 MNE EGI reader) ──
        t1 = time.time()
        raw = mne.io.read_raw_egi(args.mff, preload=False, verbose=False)
        sfreq = int(raw.info['sfreq'])
        print(f"[N{night}] MNE header: {len(raw.ch_names)} 通道, {sfreq} Hz "
              f"({time.time()-t1:.1f}s)")

        # ── 2. 选取 EEG + EOG 双通道 ──
        eeg_exists = eeg_ch in raw.ch_names
        eog_exists = eog_ch in raw.ch_names

        if not eeg_exists:
            print(f"[N{night}] ✗ EEG 通道 {eeg_ch} 不存在!")
            return 1
        if not eog_exists:
            print(f"[N{night}] ⚠ EOG 通道 {eog_ch} 不存在，仅用 EEG")
            eog_ch = None

        pick_list = [eeg_ch]
        if eog_ch:
            pick_list.append(eog_ch)

        t2 = time.time()
        raw.pick(pick_list)
        raw.load_data()
        duration_h = raw.n_times / sfreq / 3600
        print(f"[N{night}] 数据加载: {len(pick_list)} 通道, "
              f"{raw.n_times} samples, {duration_h:.1f}h "
              f"({time.time()-t2:.1f}s)")

        # ── 3. YASA 睡眠分期 ──
        t3 = time.time()
        yasa_kwargs = dict(eeg_name=eeg_ch)
        if eog_ch:
            yasa_kwargs['eog_name'] = eog_ch
            print(f"[N{night}] YASA: EEG={eeg_ch} + EOG={eog_ch}")
        else:
            print(f"[N{night}] YASA: EEG={eeg_ch} (无 EOG)")

        sls = yasa.SleepStaging(raw, **yasa_kwargs)
        print(f"[N{night}]   SleepStaging.__init__ {time.time()-t3:.1f}s")

        t4 = time.time()
        hypno = np.asarray(sls.predict(), dtype=int)
        print(f"[N{night}]   predict() {time.time()-t4:.1f}s")

        del sls, raw

        # ── 4. 输出结果 ──
        stages, counts = np.unique(hypno, return_counts=True)
        labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

        print(f"\n[N{night}] ===== 分期结果 =====")
        print(f"  总 epoch: {len(hypno)}")
        for s, c in zip(stages, counts):
            print(f"  {labels.get(s, s):>5}: {c:>4} ({c/len(hypno)*100:5.1f}%)")
        wake_count = int((hypno == 0).sum())
        print(f"  睡眠效率: {(len(hypno)-wake_count)/len(hypno)*100:.1f}%")

        # ── 5. 保存 ──
        npy_path = out_dir / f"night{night}_hypno.npy"
        np.save(str(npy_path), hypno)
        print(f"[N{night}] 保存: {npy_path}")

        elapsed = time.time() - t0
        print(f"[N{night}] ✓ Step 2 完成 ({elapsed:.1f}s)\n")
        return 0

    except Exception as e:
        print(f"[N{night}] ✗ Step 2 失败 ({time.time()-t0:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == '__main__':
    sys.exit(main())
