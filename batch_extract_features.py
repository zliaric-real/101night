# -*- coding: utf-8 -*-
"""
batch_extract_features.py — 批量提取 Step 1+2 特征（单通道 + YASA + SSSM）
====================================================================
仅提取单通道特征和睡眠分期，跳过 Step 3（连接性/微状态/空间）。
统一输出 30s epoch 级别特征。

关键处理：
  1. 多文件夜晚拼接（如 Nathalie-27a/27b/27c），按采集时间排序
  2. SSSM 特征波 3s→30s 聚合：每 epoch 统计 7 类波各出现次数
  3. 间隙填充：特征=NaN，分期=Wake(0)

用法:
  conda activate eeg_101night
  python batch_extract_features.py                        # 全部夜晚
  python batch_extract_features.py --nights 27,40,41      # 指定夜晚
  python batch_extract_features.py --nights 27 --dry-run  # 预览不执行
"""

import os as _os
_os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
_os.environ.setdefault('OMP_NUM_THREADS', '1')
_os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys, re, gc, time, warnings, argparse, traceback
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── 路径 ──
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("I:/101Night")
OUTPUT_DIR = PROJECT_DIR / "features_batch"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_PATH = PROJECT_DIR / "batch_extract_log.txt"
PROGRESS_PATH = PROJECT_DIR / "batch_extract_progress.pkl"  # 断点续跑

# ── 常量 ──
WAVE_NAMES = {
    0: "Background", 1: "Spindle", 2: "Slow wave",
    3: "K-complex", 4: "Sawtooth", 5: "Vertex sharp", 6: "Arousal"
}
WAVE_LABELS = [
    "sssm_Background", "sssm_Spindle", "sssm_Slow wave",
    "sssm_K-complex", "sssm_Sawtooth", "sssm_Vertex sharp", "sssm_Arousal"
]

# Step 1 特征组（排除 connectivity/microstates/spatial，这些属于 Step 3）
STEP1_GROUPS = (
    'basic', 'time_domain', 'frequency', 'entropy', 'complexity',
    'tsfresh', 'rqa', 'adv_spectral', 'autocorrelation'
)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


# ═══════════════════════════════════════════════════════════════
#  扫描：支持多文件夜晚 (27a, 27b, 27c)
# ═══════════════════════════════════════════════════════════════

def scan_mff_dirs(data_dir: Path = DATA_DIR):
    """扫描全部 .mff 目录，支持可选字母后缀（多文件夜晚）。

    正则匹配示例:
      Nathalie-27_20170928_121143.mff     → night=27, suffix=""
      Nathalie-27a_20170928_121143.mff    → night=27, suffix="a"
      Nathalie-27b_20170928_015108.mff    → night=27, suffix="b"
    """
    mff_dirs = sorted(data_dir.glob("Nathalie-*.mff"))
    records = []
    pat = re.compile(
        r"Nathalie-(\d+)([a-z]*)_"            # night + optional letter suffix
        r"(\d{4})(\d{2})(\d{2})_"             # YYYYMMDD
        r"(\d{2})(\d{2})(\d{2})"              # HHMMSS
        r"\.mff"
    )
    skipped = 0
    for mff_path in mff_dirs:
        name = mff_path.name
        m = pat.match(name)
        if m:
            night = int(m.group(1))
            suffix = m.group(2) or ""
            y, mo, d = int(m.group(3)), int(m.group(4)), int(m.group(5))
            hh, mm, ss = int(m.group(6)), int(m.group(7)), int(m.group(8))
            dt = datetime(y, mo, d, hh, mm, ss)
            records.append({
                "night": night,
                "suffix": suffix,
                "datetime": dt,
                "path": str(mff_path),
                "name": name,
            })
        else:
            skipped += 1

    if skipped:
        log(f"[Scan] ⚠ {skipped} 个目录名称不匹配正则, 已跳过")

    records.sort(key=lambda r: (r["night"], r["datetime"]))
    return records


def group_by_night(records: list) -> dict:
    """按 night 编号分组，组内按 datetime 排序。"""
    groups = {}
    for rec in records:
        night = rec["night"]
        groups.setdefault(night, []).append(rec)
    return dict(sorted(groups.items()))


# ═══════════════════════════════════════════════════════════════
#  SSSM 3s → 30s 聚合
# ═══════════════════════════════════════════════════════════════

def aggregate_sssm_to_epochs(pred_labels, n_epochs: int) -> np.ndarray:
    """将 SSSM 特征波（3秒窗口）聚合到 30秒 epoch。

    SSSM: 100 Hz, step=300 samples → 每 3 秒一个窗口
    每个 30s epoch ≈ 10 个 SSSM 窗口

    Args:
        pred_labels: SSSM 模型输出的标签数组, shape (1, n_windows) 或 (n_windows,)
        n_epochs: 30秒 epoch 数量

    Returns:
        np.ndarray, shape (n_epochs, 7), dtype=int
        7 列依次对应: Background, Spindle, Slow wave, K-complex,
                     Sawtooth, Vertex sharp, Arousal
    """
    pred_labels = np.asarray(pred_labels)
    if pred_labels.ndim > 1:
        pred_labels = pred_labels.flatten()

    n_windows = len(pred_labels)
    if n_epochs <= 0 or n_windows == 0:
        return np.zeros((max(n_epochs, 0), 7), dtype=int)

    wave_counts = np.zeros((n_epochs, 7), dtype=int)
    windows_per_epoch = n_windows / n_epochs

    for ep in range(n_epochs):
        start = int(np.round(ep * windows_per_epoch))
        end = int(np.round((ep + 1) * windows_per_epoch))
        start = max(0, min(start, n_windows))
        end = max(start, min(end, n_windows))
        if end > start:
            seg = pred_labels[start:end]
            for cls_id in range(7):
                wave_counts[ep, cls_id] = int(np.sum(seg == cls_id))

    return wave_counts


# ═══════════════════════════════════════════════════════════════
#  单文件处理
# ═══════════════════════════════════════════════════════════════

def process_one_file(rec: dict) -> dict | None:
    """处理单个 .mff 文件，提取 Step 1+2 特征。

    Returns:
        dict with keys:
          - df: pd.DataFrame (每行一个 30s epoch)
          - n_times: int (原始采样点数)
          - sfreq: float
          - ok: bool
          - error: str|None
        失败时返回 None
    """
    night = rec["night"]
    suffix = rec["suffix"]
    mff_path = rec["path"]
    label = f"N{night}{'_' + suffix if suffix else ''}"

    log(f"  [{label}] 开始处理: {rec['name']}")

    try:
        # 延迟导入，避免顶层 import 拖慢扫描
        import importlib.util

        _spec = importlib.util.spec_from_file_location(
            "feature_101night_analy",
            str(PROJECT_DIR / "feature_101night_analy.py")
        )
        _analy = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_analy)
        SleepEEGFeatureExtractor = _analy.SleepEEGFeatureExtractor

        t0 = time.time()

        # ── 初始化 (load_all_channels=False 跳过半球通道预滤波) ──
        ext = SleepEEGFeatureExtractor(
            mff_path,
            eeg_channel='E21',
            eog_channel='E61',
            load_all_channels=False,   # 只加载 E21+E61, 跳过半球通道
            max_per_hemi=5,
        )
        t_init = time.time() - t0
        n_times = len(ext.data)
        sfreq = ext.sfreq
        duration_h = n_times / sfreq / 3600
        n_epochs_expected = int(n_times // (30 * sfreq))
        log(f"  [{label}] 初始化 {t_init:.0f}s, {sfreq:.0f}Hz, "
            f"{duration_h:.1f}h, ~{n_epochs_expected} epochs")

        # ── 运行 Step 1+2 (自动跳过 Step 3) ──
        t_run = time.time()
        features = ext.run_all(
            epoch_sec=30,
            groups=STEP1_GROUPS,          # 只含单通道特征组
            skip_yasa=False,
            skip_sssm=False,
            skip_source_loc=True,          # 跳过源定位
        )
        dt_run = time.time() - t_run
        log(f"  [{label}] 特征提取 {dt_run:.0f}s")

        # ── 导出 DataFrame ──
        df = ext.to_dataframe(epoch_sec=30)
        n_epochs = len(df)
        log(f"  [{label}] DataFrame: {n_epochs} epochs, {len(df.columns)} 列")

        # ── 提取睡眠分期 ──
        if 'sleep_stage' in ext.features:
            stages = np.asarray(ext.features['sleep_stage'], dtype=int)
            if len(stages) == n_epochs:
                df['sleep_stage'] = stages
            elif len(stages) > n_epochs:
                df['sleep_stage'] = stages[:n_epochs]
                log(f"  [{label}] ⚠ 睡眠分期比 epoch 多 "
                    f"({len(stages)} vs {n_epochs}), 截断")
            else:
                df['sleep_stage'] = -1
                df.loc[:len(stages)-1, 'sleep_stage'] = stages
                log(f"  [{label}] ⚠ 睡眠分期比 epoch 少 "
                    f"({len(stages)} vs {n_epochs}), 尾部填 -1")
        else:
            log(f"  [{label}] ⚠ 无睡眠分期数据")
            df['sleep_stage'] = -1

        # ── 提取并聚合 SSSM 特征波 ──
        sssm_ok = False
        if 'feature_waves' in ext.features:
            pred_labels = np.asarray(ext.features['feature_waves'])
            sssm_counts = aggregate_sssm_to_epochs(pred_labels, n_epochs)
            for i, label_name in enumerate(WAVE_LABELS):
                df[label_name] = sssm_counts[:, i]
            sssm_ok = True
            total_waves = sssm_counts.sum()
            log(f"  [{label}] SSSM: {pred_labels.size} 窗口 → "
                f"{n_epochs} epochs, 总波数={total_waves}")
        else:
            log(f"  [{label}] ⚠ 无 SSSM 特征波数据")
            for label_name in WAVE_LABELS:
                df[label_name] = 0

        # ── 添加元数据列 ──
        df['night'] = night
        df['file_segment'] = suffix if suffix else 'main'
        df['file_start'] = rec['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        df['file_duration_h'] = duration_h

        # ── 清理内存 ──
        del ext, _analy, SleepEEGFeatureExtractor
        gc.collect()

        elapsed = time.time() - t0
        log(f"  [{label}] ✓ 完成 ({elapsed:.0f}s)")

        return {
            'df': df,
            'n_times': n_times,
            'sfreq': sfreq,
            'n_epochs': n_epochs,
            'duration_h': duration_h,
            'sssm_ok': sssm_ok,
            'ok': True,
            'error': None,
        }

    except Exception as e:
        log(f"  [{label}] ✗ 失败: {e}")
        traceback.print_exc()
        gc.collect()
        return None


# ═══════════════════════════════════════════════════════════════
#  单夜拼接（多文件 + 间隙填充）
# ═══════════════════════════════════════════════════════════════

def _build_gap_epochs(n_gap: int, night: int) -> pd.DataFrame:
    """构建间隙 epoch DataFrame，特征全 NaN，分期=Wake(0)。"""
    rows = []
    for i in range(n_gap):
        row = {
            'epoch': -1,           # 稍后重新编号
            'sleep_stage': 0,      # Wake
            'night': night,
            'file_segment': 'gap',
            'file_start': '',
            'file_duration_h': 0.0,
        }
        for wl in WAVE_LABELS:
            row[wl] = 0
        rows.append(row)
    # 用 NaN 填充所有特征列（后续 concat 时会自动对齐）
    df_gap = pd.DataFrame(rows)
    return df_gap


def process_one_night(records: list) -> dict | None:
    """处理一个夜晚（可能多个文件）。

    按采集时间排序后依次处理各文件，间隙插入 Wake epoch。

    Returns:
        dict with keys: night, n_files, n_epochs, n_gap_epochs, df, ok
    """
    night = records[0]["night"]
    n_files = len(records)
    label_parts = []
    for r in records:
        s = r['suffix']
        label_parts.append(s if s else 'main')
    night_label = f"N{night} ({','.join(label_parts)})"

    log(f"\n{'='*60}")
    log(f"Night {night}: {n_files} 文件 — {night_label}")
    log(f"{'='*60}")

    file_results = []
    for i, rec in enumerate(records):
        log(f"  [{i+1}/{n_files}] {rec['name']} "
            f"({rec['datetime'].strftime('%Y-%m-%d %H:%M:%S')})")
        result = process_one_file(rec)
        if result is not None and result['ok']:
            file_results.append(result)
        else:
            log(f"  [{i+1}/{n_files}] ✗ 文件处理失败，跳过")
            # 继续处理其他文件

    if not file_results:
        log(f"Night {night}: 所有文件处理失败！")
        return None

    # ── 计算间隙并拼接 ──
    dfs = []
    total_gap_epochs = 0
    gap_details = []

    for i, fres in enumerate(file_results):
        df = fres['df'].copy()
        rec = records[i]
        start_dt = rec['datetime']
        duration_h = fres['duration_h']
        end_dt = start_dt + timedelta(seconds=duration_h * 3600)

        # 计算与前一个文件的间隙
        if i > 0:
            prev_rec = records[i - 1]
            prev_fres = file_results[i - 1]
            prev_end = prev_rec['datetime'] + timedelta(
                seconds=prev_fres['duration_h'] * 3600
            )
            gap_sec = (start_dt - prev_end).total_seconds()

            if gap_sec > 0:
                # 有间隙：插入 Wake epoch
                n_gap = int(np.ceil(gap_sec / 30))
                df_gap = _build_gap_epochs(n_gap, night)
                dfs.append(df_gap)
                total_gap_epochs += n_gap
                gap_details.append(
                    f"  [{rec['suffix'] or 'main'}] 前间隙: "
                    f"{gap_sec/60:.1f}min → {n_gap} epochs (Wake)"
                )
                log(gap_details[-1])
            elif gap_sec < -30:
                # 重叠 > 30s：警告但继续
                log(f"  [{rec['suffix'] or 'main'}] ⚠ 与前一文件重叠 "
                    f"{-gap_sec/60:.1f}min")

        dfs.append(df)

    # 拼接全部
    df_all = pd.concat(dfs, ignore_index=True)

    # 重新编号 epoch
    df_all['epoch'] = range(len(df_all))

    # 对 gap 行，确保所有特征列为 NaN（sleep_stage 和 sssm 已设为 0）
    gap_mask = df_all['file_segment'] == 'gap'
    if gap_mask.any():
        # 找出所有非元数据/非标签列 → 设为 NaN
        meta_cols = {'epoch', 'night', 'file_segment', 'file_start',
                     'file_duration_h', 'sleep_stage'}
        meta_cols.update(WAVE_LABELS)  # SSSM 保持 0
        feature_cols = [c for c in df_all.columns if c not in meta_cols]
        df_all.loc[gap_mask, feature_cols] = np.nan

    n_total = len(df_all)
    n_real = n_total - total_gap_epochs
    log(f"Night {night}: {n_total} total epochs "
        f"({n_real} real + {total_gap_epochs} gap)")

    # ── 保存 ──
    csv_path = OUTPUT_DIR / f"features_night{night}.csv"
    df_all.to_csv(csv_path, index=False, float_format='%.6g')
    log(f"  → 已保存: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    # ── 清理 ──
    for fres in file_results:
        del fres['df']
    gc.collect()

    return {
        'night': night,
        'n_files': n_files,
        'n_epochs': n_total,
        'n_real_epochs': n_real,
        'n_gap_epochs': total_gap_epochs,
        'gap_details': gap_details,
        'csv_path': str(csv_path),
        'ok': True,
    }


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='批量提取 Step 1+2 特征（单通道 + YASA + SSSM）')
    parser.add_argument('--nights', default=None,
                        help='指定夜晚编号, 逗号分隔 (如 27,40,41)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅扫描预览, 不执行提取')
    parser.add_argument('--resume', action='store_true',
                        help='从断点续跑 (跳过已完成夜晚)')
    args = parser.parse_args()

    # 清空日志
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    log("=" * 60)
    log("batch_extract_features.py — Step 1+2 批量特征提取")
    log(f"数据目录: {DATA_DIR}")
    log(f"输出目录: {OUTPUT_DIR}")
    log("=" * 60)

    # ── 扫描 ──
    records = scan_mff_dirs()
    if not records:
        log("✗ 未找到任何 .mff 目录！请检查 DATA_DIR 路径。")
        sys.exit(1)

    nights_map = group_by_night(records)
    all_nights = sorted(nights_map.keys())
    log(f"\n扫描到 {len(records)} 个文件, 共 {len(all_nights)} 个夜晚")

    # 打印多文件夜晚
    multi_file_nights = {n: recs for n, recs in nights_map.items()
                         if len(recs) > 1}
    if multi_file_nights:
        log(f"\n多文件夜晚 ({len(multi_file_nights)} 个):")
        for n, recs in sorted(multi_file_nights.items()):
            parts = [f"{r['suffix'] or 'main'}:{r['datetime'].strftime('%H:%M')}"
                     for r in recs]
            log(f"  Night {n}: {', '.join(parts)}")

    # ── 筛选 ──
    if args.nights:
        target = {int(n.strip()) for n in args.nights.split(',')}
        nights_map = {n: recs for n, recs in nights_map.items()
                      if n in target}
        if not nights_map:
            log(f"✗ 指定夜晚未找到: {args.nights}")
            sys.exit(1)

    target_nights = sorted(nights_map.keys())
    log(f"\n目标夜晚: {target_nights}")

    if args.dry_run:
        log("\n[Dry-run] 预览完成，不执行提取。")
        for n in target_nights:
            recs = nights_map[n]
            for r in recs:
                log(f"  N{n}{'_'+r['suffix'] if r['suffix'] else ''}: "
                    f"{r['datetime'].strftime('%Y-%m-%d %H:%M:%S')} "
                    f"→ {r['name']}")
        return

    # ── 断点续跑 ──
    done_nights = set()
    if args.resume and PROGRESS_PATH.exists():
        import pickle
        with open(PROGRESS_PATH, 'rb') as f:
            progress = pickle.load(f)
        done_nights = set(progress.get('done', []))
        log(f"\n断点续跑: 已完成 {len(done_nights)} 个夜晚")

    # ── 逐夜处理 ──
    results = []
    for night in target_nights:
        if night in done_nights:
            log(f"\nNight {night}: 已完成, 跳过")
            continue

        recs = nights_map[night]
        result = process_one_night(recs)
        if result:
            results.append(result)

        # 断点保存
        done_nights.add(night)
        import pickle
        with open(PROGRESS_PATH, 'wb') as f:
            pickle.dump({'done': list(done_nights), 'results': results}, f)

    # ── 汇总 ──
    ok_count = sum(1 for r in results if r and r.get('ok'))
    total_epochs = sum(r['n_epochs'] for r in results if r)
    total_real = sum(r['n_real_epochs'] for r in results if r)
    total_gap = sum(r['n_gap_epochs'] for r in results if r)

    log(f"\n{'='*60}")
    log(f"批处理完成!")
    log(f"  成功: {ok_count}/{len(target_nights)} 夜晚")
    log(f"  总 epochs: {total_epochs} ({total_real} real + {total_gap} gap)")
    log(f"  输出目录: {OUTPUT_DIR}")
    log(f"  日志: {LOG_PATH}")
    log(f"{'='*60}")

    # 打印各夜统计
    log(f"\n各夜统计:")
    for r in results:
        if r:
            log(f"  N{r['night']:3d}: {r['n_epochs']:5d} epochs "
                f"({r['n_real_epochs']} real + {r['n_gap_epochs']} gap) "
                f"→ {Path(r['csv_path']).name}")


if __name__ == "__main__":
    main()
