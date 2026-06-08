# -*- coding: utf-8 -*-
"""
batch_extract_features.py — 批量提取 Step 1+2 特征（单通道 + YASA + SSSM）
====================================================================
仅提取单通道特征和睡眠分期，跳过 Step 3（连接性/微状态/空间）。
统一输出 30s epoch 级别特征。

设计为"中转脚本"：可被 Jupyter notebook 直接 import 调用，
也可通过 CLI 独立运行。

关键处理：
  1. 双数据源：支持多个目录同时扫描，自动去重
  2. 多文件夜晚拼接（如 Nathalie-27a/27b/27c），按采集时间排序
  3. SSSM 特征波 3s→30s 聚合：每 epoch 统计 7 类波各出现次数
  4. 间隙填充：特征=NaN，分期=Wake(0)

═══════════════════════════════════════════════════════════════
 Jupyter Notebook 调用示例
═══════════════════════════════════════════════════════════════

  from batch_extract_features import (
      scan_mff_dirs, group_by_night,
      process_one_night, run_batch,
      DEFAULT_DATA_DIRS, set_log_path,
  )

  # 1. 扫描全部数据源
  records = scan_mff_dirs()
  nights = group_by_night(records)
  print(f"发现 {len(nights)} 个夜晚")

  # 2. 处理单个夜晚，直接获取 DataFrame
  result = process_one_night(nights[27])
  df = result['df']          # pd.DataFrame, 每行一个 30s epoch
  print(df.head())

  # 3. 批量处理，逐夜获取结果
  for result in run_batch(nights, target=[27, 40, 41]):
      df = result['df']
      # 自定义分析...
      print(f"Night {result['night']}: {len(df)} epochs")

  # 4. 自定义日志路径
  set_log_path("my_log.txt")

═══════════════════════════════════════════════════════════════
 CLI 用法
═══════════════════════════════════════════════════════════════

  conda activate eeg_101night
  python batch_extract_features.py --dry-run
  python batch_extract_features.py --nights 27,40,41
  python batch_extract_features.py --resume
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

# ═══════════════════════════════════════════════════════════════
#  可配置路径 — notebook 可通过 set_* 函数修改
# ═══════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_DIRS = [
    Path("I:/101Night"),
    Path("E:/zhaochenhao/data/101Night"),
]

_OUTPUT_DIR = PROJECT_DIR / "features_batch"
_LOG_PATH = PROJECT_DIR / "batch_extract_log.txt"
_PROGRESS_PATH = PROJECT_DIR / "batch_extract_progress.pkl"


def set_output_dir(path):
    """设置特征 CSV 输出目录（notebook 调用）。"""
    global _OUTPUT_DIR
    _OUTPUT_DIR = Path(path)
    _OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def set_log_path(path):
    """设置日志文件路径（notebook 调用）。"""
    global _LOG_PATH
    _LOG_PATH = Path(path)


def set_progress_path(path):
    """设置断点续跑进度文件路径。"""
    global _PROGRESS_PATH
    _PROGRESS_PATH = Path(path)


# ═══════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════

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

_MFF_PATTERN = re.compile(
    r"Nathalie-(\d+)([a-z]*)_"            # night + optional letter suffix
    r"(\d{4})(\d{2})(\d{2})_"             # YYYYMMDD
    r"(\d{2})(\d{2})(\d{2})"              # HHMMSS
    r"\.mff"
)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass  # notebook 环境可能无法写文件，忽略


# ═══════════════════════════════════════════════════════════════
#  扫描：多数据源 + 去重 + 支持多文件夜晚
# ═══════════════════════════════════════════════════════════════

def scan_mff_dirs(data_dirs=None):
    """扫描多个目录中的全部 .mff 文件。

    自动去重：同一 (night, suffix, datetime) 的文件只保留第一个。
    支持可选字母后缀（多文件夜晚）:
      Nathalie-27_20170928_121143.mff     → night=27, suffix=""
      Nathalie-27a_20170928_121143.mff    → night=27, suffix="a"

    Args:
        data_dirs: 数据目录列表，默认 DEFAULT_DATA_DIRS

    Returns:
        list[dict]: 每项包含 night, suffix, datetime, path, name, source
    """
    if data_dirs is None:
        data_dirs = DEFAULT_DATA_DIRS
    elif isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]

    records = []
    seen = set()   # (night, suffix, datetime) 去重
    skipped_no_match = 0
    skipped_dup = 0

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            log(f"[Scan] ⚠ 目录不存在, 跳过: {data_dir}")
            continue

        mff_dirs = sorted(data_dir.glob("Nathalie-*.mff"))
        log(f"[Scan] {data_dir}: 发现 {len(mff_dirs)} 个 .mff 目录")

        for mff_path in mff_dirs:
            name = mff_path.name
            m = _MFF_PATTERN.match(name)
            if not m:
                skipped_no_match += 1
                continue

            night = int(m.group(1))
            suffix = m.group(2) or ""
            y, mo, d = int(m.group(3)), int(m.group(4)), int(m.group(5))
            hh, mm, ss = int(m.group(6)), int(m.group(7)), int(m.group(8))
            dt = datetime(y, mo, d, hh, mm, ss)

            key = (night, suffix, dt)
            if key in seen:
                skipped_dup += 1
                continue
            seen.add(key)

            records.append({
                "night": night,
                "suffix": suffix,
                "datetime": dt,
                "path": str(mff_path),
                "name": name,
                "source": str(data_dir),     # 来源目录，便于追溯
            })

    if skipped_no_match:
        log(f"[Scan] ⚠ {skipped_no_match} 个目录名称不匹配正则, 已跳过")
    if skipped_dup:
        log(f"[Scan] ⚠ {skipped_dup} 个重复 (night+suffix+datetime), 已去重")

    records.sort(key=lambda r: (r["night"], r["datetime"]))
    log(f"[Scan] 总计: {len(records)} 个有效文件")
    return records


def group_by_night(records: list) -> dict:
    """按 night 编号分组，组内按 datetime 排序。

    Returns:
        dict: {night: [record, ...]}
    """
    groups = {}
    for rec in records:
        night = rec["night"]
        groups.setdefault(night, []).append(rec)
    # 组内已按 scan 时的全局排序，但确保一下
    for recs in groups.values():
        recs.sort(key=lambda r: r["datetime"])
    return dict(sorted(groups.items()))


def summarize_nights(records: list):
    """打印夜晚摘要（notebook 友好）。"""
    nights = group_by_night(records)
    print(f"\n{'='*60}")
    print(f"扫描摘要: {len(records)} 文件, {len(nights)} 个夜晚")
    print(f"{'='*60}")

    multi = {n: rs for n, rs in nights.items() if len(rs) > 1}
    if multi:
        print(f"\n多文件夜晚 ({len(multi)} 个):")
        for n, rs in sorted(multi.items()):
            parts = [f"{r['suffix'] or 'main'} ({r['datetime'].strftime('%m/%d %H:%M')})"
                     for r in rs]
            print(f"  Night {n}: {'  →  '.join(parts)}")

    single = [n for n, rs in nights.items() if len(rs) == 1]
    print(f"\n单文件夜晚: {len(single)} 个")


# ═══════════════════════════════════════════════════════════════
#  SSSM 3s → 30s 聚合
# ═══════════════════════════════════════════════════════════════

def aggregate_sssm_to_epochs(pred_labels, n_epochs: int) -> np.ndarray:
    """将 SSSM 特征波（3秒窗口）聚合到 30秒 epoch。

    SSSM: 100 Hz, step=300 samples → 每 3 秒一个窗口
    每个 30s epoch ≈ 10 个 SSSM 窗口
    处理非整数比例（如 19 窗口 ÷ 2 epoch），按比例分配

    Args:
        pred_labels: SSSM 模型输出的标签数组, shape (1, n_windows) 或 (n_windows,)
        n_epochs: 30秒 epoch 数量

    Returns:
        np.ndarray, shape (n_epochs, 7), dtype=int
        列: Background, Spindle, Slow wave, K-complex, Sawtooth, Vertex sharp, Arousal
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

def _load_extractor_class():
    """延迟加载 SleepEEGFeatureExtractor（避免 import 耗时影响扫描）。"""
    import importlib.util
    spec_path = str(PROJECT_DIR / "feature_101night_analy.py")
    # 每次调用使用唯一 name 避免模块缓存冲突
    name = f"feature_101night_analy_{id(_load_extractor_class)}"
    spec = importlib.util.spec_from_file_location(name, spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SleepEEGFeatureExtractor


def process_one_file(rec: dict) -> dict | None:
    """处理单个 .mff 文件，提取 Step 1+2 特征。

    仅加载 E21+E61 通道（~60 MB），跳过半球通道预加载。
    返回的 DataFrame 每行对应一个 30s epoch。

    Args:
        rec: scan_mff_dirs() 返回的单条记录

    Returns:
        dict: {
            'df': pd.DataFrame,   # 每行一个 30s epoch
            'n_times': int,       # 原始采样点数
            'sfreq': float,       # 采样率
            'n_epochs': int,      # epoch 数
            'duration_h': float,  # 时长(小时)
            'sssm_ok': bool,      # SSSM 是否成功
            'ok': bool,
            'error': str | None,
        }
        失败时返回 None
    """
    night = rec["night"]
    suffix = rec["suffix"]
    mff_path = rec["path"]
    label = f"N{night}{'_' + suffix if suffix else ''}"

    log(f"  [{label}] 开始: {rec['name']}")

    try:
        SleepEEGFeatureExtractor = _load_extractor_class()
        t0 = time.time()

        # ── 初始化 (load_all_channels=False → 只加载 E21+E61) ──
        ext = SleepEEGFeatureExtractor(
            mff_path,
            eeg_channel='E21',
            eog_channel='E61',
            load_all_channels=False,
            max_per_hemi=5,
        )
        t_init = time.time() - t0
        n_times = len(ext.data)
        sfreq = ext.sfreq
        duration_h = n_times / sfreq / 3600
        n_epochs_expected = int(n_times // (30 * sfreq))
        log(f"  [{label}] init {t_init:.0f}s, {sfreq:.0f}Hz, "
            f"{duration_h:.1f}h, ~{n_epochs_expected} epochs")

        # ── Step 1+2 ──
        t_run = time.time()
        ext.run_all(
            epoch_sec=30,
            groups=STEP1_GROUPS,
            skip_yasa=False,
            skip_sssm=False,
            skip_source_loc=True,
        )
        dt_run = time.time() - t_run
        log(f"  [{label}] 特征提取 {dt_run:.0f}s")

        # ── DataFrame ──
        df = ext.to_dataframe(epoch_sec=30)
        n_epochs = len(df)

        # ── 睡眠分期 ──
        if 'sleep_stage' in ext.features:
            stages = np.asarray(ext.features['sleep_stage'], dtype=int)
            if len(stages) == n_epochs:
                df['sleep_stage'] = stages
            elif len(stages) > n_epochs:
                df['sleep_stage'] = stages[:n_epochs]
                log(f"  [{label}] ⚠ 分期({len(stages)}) > epoch({n_epochs}), 截断")
            else:
                df['sleep_stage'] = -1
                df.loc[:len(stages)-1, 'sleep_stage'] = stages
                log(f"  [{label}] ⚠ 分期({len(stages)}) < epoch({n_epochs}), 尾部填-1")
        else:
            df['sleep_stage'] = -1

        # ── SSSM 聚合 ──
        sssm_ok = False
        if 'feature_waves' in ext.features:
            pred = np.asarray(ext.features['feature_waves'])
            sssm_counts = aggregate_sssm_to_epochs(pred, n_epochs)
            for i, lbl in enumerate(WAVE_LABELS):
                df[lbl] = sssm_counts[:, i]
            sssm_ok = True
            log(f"  [{label}] SSSM: {pred.size}窗 → {n_epochs}ep, "
                f"波数={sssm_counts.sum()}")
        else:
            for lbl in WAVE_LABELS:
                df[lbl] = 0

        # ── 元数据 ──
        df['night'] = night
        df['file_segment'] = suffix if suffix else 'main'
        df['file_start'] = rec['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        df['file_duration_h'] = duration_h
        df['source'] = rec.get('source', '')

        del ext
        gc.collect()

        elapsed = time.time() - t0
        log(f"  [{label}] ✓ {elapsed:.0f}s, {n_epochs} epochs, "
            f"{len(df.columns)} 列")

        return {
            'df': df, 'n_times': n_times, 'sfreq': sfreq,
            'n_epochs': n_epochs, 'duration_h': duration_h,
            'sssm_ok': sssm_ok, 'ok': True, 'error': None,
        }

    except Exception as e:
        log(f"  [{label}] ✗ 失败: {e}")
        traceback.print_exc()
        gc.collect()
        return None


# ═══════════════════════════════════════════════════════════════
#  间隙填充
# ═══════════════════════════════════════════════════════════════

def _build_gap_rows(n_gap: int, night: int) -> list[dict]:
    """构建间隙 epoch 行（用于拼接）。"""
    rows = []
    for _ in range(n_gap):
        row = {
            'epoch': -1, 'sleep_stage': 0, 'night': night,
            'file_segment': 'gap', 'file_start': '',
            'file_duration_h': 0.0, 'source': '',
        }
        for wl in WAVE_LABELS:
            row[wl] = 0
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════
#  单夜拼接（多文件 + 间隙填充）
# ═══════════════════════════════════════════════════════════════

def process_one_night(records: list,
                      save_csv: bool = True,
                      output_dir=None) -> dict | None:
    """处理一个夜晚（可能多个文件），拼接并填充间隙。

    按采集时间排序后依次处理各文件，文件间间隙插入 Wake epoch。
    可作为 notebook API — 调用后 result['df'] 直接拿到完整 DataFrame。

    Args:
        records: 该夜晚的全部文件记录（已按 datetime 排序）
        save_csv: 是否保存 CSV（notebook 中可设为 False）
        output_dir: 输出目录，默认 _OUTPUT_DIR

    Returns:
        dict: {
            'night': int,
            'n_files': int,
            'n_epochs': int,         # 含间隙的总 epoch 数
            'n_real_epochs': int,    # 实际数据 epoch 数
            'n_gap_epochs': int,     # 间隙填充 epoch 数
            'df': pd.DataFrame,      # 完整拼接后的 DataFrame
            'csv_path': str | None,  # CSV 路径（如果保存了）
            'gap_details': list[str],
            'ok': bool,
        }
        所有文件失败时返回 None
    """
    night = records[0]["night"]
    n_files = len(records)
    out_dir = Path(output_dir) if output_dir else _OUTPUT_DIR
    out_dir.mkdir(exist_ok=True, parents=True)

    parts = [r['suffix'] or 'main' for r in records]
    log(f"\n{'='*60}")
    log(f"Night {night}: {n_files} 文件 ({','.join(parts)})")
    log(f"{'='*60}")

    # ── 逐文件处理 ──
    file_results = []
    for i, rec in enumerate(records):
        log(f"  [{i+1}/{n_files}] {rec['name']} "
            f"({rec['datetime'].strftime('%Y-%m-%d %H:%M:%S')})")
        result = process_one_file(rec)
        if result and result['ok']:
            file_results.append(result)
        else:
            log(f"  [{i+1}/{n_files}] ✗ 跳过")

    if not file_results:
        log(f"Night {night}: 全部文件失败！")
        return None

    # ── 计算间隙 + 拼接 ──
    segments = []       # list of (df or list-of-rows, is_gap)
    total_gap = 0
    gap_details = []

    # 提取实际使用的 records（与 file_results 对应）
    used_records = [records[i] for i, r in enumerate(file_results)
                    if i < len(records)]

    for i, fres in enumerate(file_results):
        rec = used_records[i] if i < len(used_records) else records[i]

        # 间隙检测
        if i > 0:
            prev_rec = used_records[i - 1] if i - 1 < len(used_records) else records[i - 1]
            prev_fres = file_results[i - 1]
            prev_end = prev_rec['datetime'] + timedelta(
                seconds=prev_fres['duration_h'] * 3600)
            gap_sec = (rec['datetime'] - prev_end).total_seconds()

            if gap_sec > 0:
                n_gap = int(np.ceil(gap_sec / 30))
                gap_rows = _build_gap_rows(n_gap, night)
                segments.append((gap_rows, True))
                total_gap += n_gap
                detail = (f"  [{rec['suffix'] or 'main'}] 前间隙: "
                          f"{gap_sec/60:.1f}min → {n_gap} epochs")
                gap_details.append(detail)
                log(detail)
            elif gap_sec < -30:
                log(f"  [{rec['suffix'] or 'main'}] ⚠ 重叠 "
                    f"{-gap_sec/60:.1f}min")

        segments.append((fres['df'], False))

    # ── 拼接 ──
    dfs_to_concat = []
    for seg, is_gap in segments:
        if is_gap:
            dfs_to_concat.append(pd.DataFrame(seg))
        else:
            dfs_to_concat.append(seg)

    df_all = pd.concat(dfs_to_concat, ignore_index=True)
    df_all['epoch'] = range(len(df_all))

    # Gap 行的特征列 → NaN
    gap_mask = df_all['file_segment'] == 'gap'
    if gap_mask.any():
        meta_cols = {'epoch', 'night', 'file_segment', 'file_start',
                     'file_duration_h', 'sleep_stage', 'source'}
        meta_cols.update(WAVE_LABELS)
        feature_cols = [c for c in df_all.columns if c not in meta_cols]
        df_all.loc[gap_mask, feature_cols] = np.nan

    n_total = len(df_all)
    n_real = n_total - total_gap
    log(f"Night {night}: {n_total} epochs ({n_real} real + {total_gap} gap)")

    # ── 保存 CSV ──
    csv_path = None
    if save_csv:
        csv_path = out_dir / f"features_night{night}.csv"
        df_all.to_csv(csv_path, index=False, float_format='%.6g')
        log(f"  → {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    # ── 清理中间 DataFrames ──
    for fres in file_results:
        del fres['df']
    gc.collect()

    return {
        'night': night,
        'n_files': n_files,
        'n_epochs': n_total,
        'n_real_epochs': n_real,
        'n_gap_epochs': total_gap,
        'df': df_all,
        'csv_path': str(csv_path) if csv_path else None,
        'gap_details': gap_details,
        'ok': True,
    }


# ═══════════════════════════════════════════════════════════════
#  批量运行（generator — notebook 友好）
# ═══════════════════════════════════════════════════════════════

def run_batch(nights_map: dict,
              target: list | set | None = None,
              save_csv: bool = True,
              output_dir=None,
              resume: bool = False) -> dict:
    """批量处理多个夜晚，generator 逐夜返回结果。

    Args:
        nights_map: group_by_night() 的输出
        target: 指定夜晚编号列表，None = 全部
        save_csv: 每夜是否保存 CSV
        output_dir: 输出目录
        resume: 是否从断点续跑

    Yields:
        dict: 每夜的处理结果（同 process_one_night 返回格式）
    """
    if target is not None:
        if isinstance(target, (list, set, tuple)):
            nights_map = {n: rs for n, rs in nights_map.items()
                          if n in target}
        else:
            nights_map = {target: nights_map[target]}

    target_nights = sorted(nights_map.keys())

    # 断点续跑
    done = set()
    if resume and _PROGRESS_PATH.exists():
        import pickle
        try:
            with open(_PROGRESS_PATH, 'rb') as f:
                progress = pickle.load(f)
            done = set(progress.get('done', []))
            log(f"断点续跑: 已完成 {len(done)} 个夜晚")
        except Exception:
            pass

    # 逐夜处理
    for night in target_nights:
        if night in done:
            log(f"Night {night}: 已完成, 跳过")
            continue

        recs = nights_map[night]
        result = process_one_night(recs,
                                   save_csv=save_csv,
                                   output_dir=output_dir)
        if result:
            yield result

        # 保存进度
        done.add(night)
        if _PROGRESS_PATH:
            import pickle
            try:
                with open(_PROGRESS_PATH, 'wb') as f:
                    pickle.dump({'done': list(done)}, f)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='批量提取 Step 1+2 特征（单通道 + YASA + SSSM）')
    parser.add_argument('--nights', default=None,
                        help='指定夜晚编号, 逗号分隔 (如 27,40,41)')
    parser.add_argument('--data-dirs', default=None,
                        help='额外数据目录, 逗号分隔 (追加到默认目录后)')
    parser.add_argument('--output', default=None,
                        help='输出目录 (默认 features_batch/)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅扫描预览, 不执行提取')
    parser.add_argument('--resume', action='store_true',
                        help='从断点续跑')
    args = parser.parse_args()

    # 清空日志
    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    log("=" * 60)
    log("batch_extract_features.py — Step 1+2 批量特征提取")
    log("=" * 60)

    # ── 数据目录 ──
    data_dirs = list(DEFAULT_DATA_DIRS)
    if args.data_dirs:
        data_dirs.extend(Path(d.strip()) for d in args.data_dirs.split(','))
    log(f"数据目录: {[str(d) for d in data_dirs]}")

    if args.output:
        set_output_dir(args.output)
    log(f"输出目录: {_OUTPUT_DIR}")

    # ── 扫描 ──
    records = scan_mff_dirs(data_dirs)
    if not records:
        log("✗ 未找到任何 .mff 目录！")
        sys.exit(1)

    nights_map = group_by_night(records)
    summarize_nights(records)

    # ── 筛选 ──
    if args.nights:
        target = {int(n.strip()) for n in args.nights.split(',')}
    else:
        target = None

    if args.dry_run:
        log("\n[Dry-run] 预览完成。")
        return

    # ── 批量运行 ──
    results = []
    for result in run_batch(nights_map,
                            target=target,
                            save_csv=True,
                            resume=args.resume):
        results.append(result)

    # ── 汇总 ──
    ok_count = sum(1 for r in results if r and r.get('ok'))
    total_ep = sum(r['n_epochs'] for r in results if r)
    total_real = sum(r['n_real_epochs'] for r in results if r)
    total_gap = sum(r['n_gap_epochs'] for r in results if r)

    log(f"\n{'='*60}")
    log(f"完成: {ok_count}/{len(results) if target else len(nights_map)} 夜晚")
    log(f"总 epochs: {total_ep} ({total_real} real + {total_gap} gap)")
    log(f"输出: {_OUTPUT_DIR}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
