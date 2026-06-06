# -*- coding: utf-8 -*-
"""
extract_single_night.py — 单夜特征提取编排器 (subprocess 模式)
==============================================================
依次调用 4 个独立 step 脚本，每个作为子进程运行：
  step1_single_channel.py — 单通道特征 + SSSM
  step2_yasa_staging.py   — YASA 睡眠分期
  step3_multichannel.py   — 多通道 ROI 特征
  step4_source_loc.py     — 源定位 (默认跳过)

每个子进程退出 = OS 自动回收全部内存，杜绝泄漏。

用法:
  python extract_single_night.py --mff I:/101Night/Nathalie-40_...mff --night 40
  python extract_single_night.py --mff ... --night 40 --steps 1,2,3       # 只跑指定步骤
  python extract_single_night.py --mff ... --night 40 --skip-yasa         # 跳过 YASA
"""

import sys
import subprocess
import time
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable  # 使用同一 Python 解释器


def run_step(step_script, args_list, night):
    """运行单个 step 子进程，返回 (success, elapsed_sec)."""
    cmd = [PYTHON, str(PROJECT_DIR / step_script)] + args_list
    print(f"\n--- 启动 {step_script} ---")
    print(f"  {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    elapsed = time.time() - t0
    success = result.returncode == 0
    status = "✓" if success else f"✗ (exit={result.returncode})"
    print(f"\n--- {step_script} {status} ({elapsed:.0f}s) ---")
    return success, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='单夜特征提取编排器 (subprocess 模式)')
    parser.add_argument('--mff', required=True, help='.mff 目录路径')
    parser.add_argument('--night', type=int, required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--eeg-channel', default='E21')
    parser.add_argument('--eog-channel', default='E61')
    parser.add_argument('--epoch-sec', type=float, default=30)
    parser.add_argument('--max-per-hemi', type=int, default=5)
    parser.add_argument('--skip-yasa', action='store_true')
    parser.add_argument('--skip-sssm', action='store_true')
    parser.add_argument('--skip-source', action='store_true',
                        help='跳过 Step 4 源定位 (默认跳过)')
    parser.add_argument('--source-method', default='eLORETA',
                        choices=['eLORETA', 'dSPM', 'MNE'])
    parser.add_argument('--slice-sec', type=float, default=600)
    parser.add_argument('--steps', default='1,2,3',
                        help='要运行的步骤 (逗号分隔, 默认: 1,2,3)')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else PROJECT_DIR
    night = args.night
    steps_to_run = [int(s.strip()) for s in args.steps.split(',')]

    print(f"\n{'#'*60}")
    print(f"# Night {night} — 分步特征提取 (subprocess)")
    print(f"# 文件: {args.mff}")
    print(f"# 步骤: {steps_to_run}")
    print(f"{'#'*60}\n")
    t_total = time.time()

    base_args = [
        '--mff', args.mff,
        '--night', str(night),
        '--output', str(out_dir),
        '--eeg-channel', args.eeg_channel,
    ]

    results = {}

    # ── Step 1: 单通道特征 ──
    if 1 in steps_to_run:
        step1_args = base_args + ['--epoch-sec', str(args.epoch_sec)]
        if args.skip_sssm:
            step1_args.append('--skip-sssm')
        ok, elapsed = run_step('step1_single_channel.py', step1_args, night)
        results[1] = ok

    # ── Step 2: YASA 分期 ──
    if 2 in steps_to_run and not args.skip_yasa:
        step2_args = base_args + ['--eog-channel', args.eog_channel]
        ok, elapsed = run_step('step2_yasa_staging.py', step2_args, night)
        results[2] = ok
    elif 2 in steps_to_run:
        print("[Step 2] 已跳过 (--skip-yasa)")

    # ── Step 3: 多通道 ROI ──
    if 3 in steps_to_run:
        step3_args = base_args + [
            '--epoch-sec', str(args.epoch_sec),
            '--max-per-hemi', str(args.max_per_hemi)]
        ok, elapsed = run_step('step3_multichannel.py', step3_args, night)
        results[3] = ok

    # ── Step 4: 源定位 ──
    if 4 in steps_to_run and not args.skip_source:
        step4_args = base_args + [
            '--epoch-sec', str(args.epoch_sec),
            '--method', args.source_method,
            '--slice-sec', str(args.slice_sec)]
        ok, elapsed = run_step('step4_source_loc.py', step4_args, night)
        results[4] = ok
    elif 4 in steps_to_run:
        print("[Step 4] 已跳过 (--skip-source)")

    elapsed = time.time() - t_total
    ok_count = sum(results.values())
    total_count = len(results)
    print(f"\n{'#'*60}")
    print(f"# Night {night} 完成: {ok_count}/{total_count} 步骤成功 "
          f"({elapsed/60:.1f} 分钟)")
    print(f"{'#'*60}")

    return 0 if ok_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
