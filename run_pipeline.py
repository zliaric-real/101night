# -*- coding: utf-8 -*-
"""
run_pipeline.py — 带内存监控的 101night 流水线运行器
===================================================
按顺序执行: YASA分期 → 特征提取 → 自相关 → 聚类
全程监控内存，超过 80% 自动终止。

用法:
  conda activate eeg_101night
  python run_pipeline.py --nights 31,40,41
  python run_pipeline.py --nights 31,40,41,42,46
"""

import os
import sys
import gc
import time
import signal
import pickle
import threading
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── 路径 ──
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("I:/101Night")
LOG_PATH = PROJECT_DIR / "pipeline_log.txt"
PIPELINE_CACHE = PROJECT_DIR / "pipeline_results.pkl"

# ── 内存阈值 ──
MEMORY_THRESHOLD = 80.0  # 80%
MEMORY_CHECK_INTERVAL = 2.0  # 秒

# ── 默认处理的夜晚 ──
DEFAULT_NIGHTS = [31, 40, 41]


# ═══════════════════════════════════════════════════════════
#  日志
# ═══════════════════════════════════════════════════════════

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


# ═══════════════════════════════════════════════════════════
#  内存监控
# ═══════════════════════════════════════════════════════════

def get_memory_pct():
    """获取当前内存使用百分比，失败返回 -1。"""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return -1


def get_memory_info():
    """获取详细内存信息。"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / 1e9,
            "available_gb": mem.available / 1e9,
            "used_pct": mem.percent,
        }
    except ImportError:
        return {"total_gb": -1, "available_gb": -1, "used_pct": -1}


class MemoryMonitor:
    """后台线程内存监控器。超阈值时设置 alarm 标志。"""

    def __init__(self, threshold=MEMORY_THRESHOLD):
        self.threshold = threshold
        self.alarm = False
        self._thread = None
        self._stop = threading.Event()
        self.peak_pct = 0.0

    def start(self):
        self._stop.clear()
        self.alarm = False
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _monitor(self):
        while not self._stop.is_set():
            pct = get_memory_pct()
            if pct > 0:
                self.peak_pct = max(self.peak_pct, pct)
                if pct > self.threshold:
                    self.alarm = True
                    log(f"⚠ MEMORY ALARM: {pct:.1f}% > {self.threshold}% 阈值!")
            time.sleep(MEMORY_CHECK_INTERVAL)


# ═══════════════════════════════════════════════════════════
#  阶段 1: YASA 睡眠分期 (批量)
# ═══════════════════════════════════════════════════════════

def run_yasa_batch(nights, monitor):
    """对指定夜晚运行 YASA 分期。

    直接调用 batch_analyze.py 的核心逻辑，仅处理 subset。
    """
    log("=" * 60)
    log(f"阶段 1: YASA 睡眠分期 — {len(nights)} 夜 {nights}")
    log("=" * 60)

    mem = get_memory_info()
    log(f"  启动内存: {mem['used_pct']:.1f}% / {mem['total_gb']:.1f} GB")

    # 导入 batch_analyze 的函数
    sys.path.insert(0, str(PROJECT_DIR))

    if 'yasa' not in sys.modules:
        log("  YASA 预导入中... (首次需要 60-120s)")
        try:
            import yasa
            log("  YASA 导入完成")
        except ImportError:
            log("  WARNING: yasa 未安装")
            return []

    from batch_analyze import (process_one_night, scan_mff_dirs,
                                _read_mff_metadata, LOG_PATH as BATCH_LOG)
    import mne

    # 扫描所有夜晚，只保留指定的
    all_records = scan_mff_dirs()
    target_set = set(nights)
    records = [r for r in all_records if r["night"] in target_set]
    records.sort(key=lambda r: r["night"])

    if not records:
        log(f"  ERROR: 未找到指定夜晚 {nights}")
        return []

    log(f"  找到 {len(records)} 个目标夜晚:")
    for r in records:
        log(f"    N{r['night']} {r['date']} — {r['path']}")

    # 检查缓存
    if PIPELINE_CACHE.exists():
        with open(PIPELINE_CACHE, "rb") as f:
            cached = pickle.load(f)
        done_yasa = {r["night"] for r in cached.get("yasa_results", [])
                     if r.get("yasa_ok")}
        log(f"  缓存中已完成 YASA: {done_yasa}")
    else:
        cached = {"yasa_results": []}
        done_yasa = set()

    results = list(cached.get("yasa_results", []))
    for rec in records:
        # 检查内存
        pct = get_memory_pct()
        if pct > MEMORY_THRESHOLD or monitor.alarm:
            log(f"  ⚠ 内存 {pct:.1f}% > {MEMORY_THRESHOLD}%，终止 YASA 批处理")
            break

        if rec["night"] in done_yasa:
            log(f"  N{rec['night']} — SKIP (已缓存)")
            continue

        log(f"\n  [{rec['night']}] {rec['date']}")
        t0 = time.time()
        result = process_one_night(rec)

        # 确保结果包含 night 字段
        if "night" not in result:
            result["night"] = rec["night"]

        results.append(result)
        elapsed = time.time() - t0

        if result.get("yasa_ok"):
            log(f"    ✓ 成功 ({elapsed:.0f}s) — {result.get('stage_pct', '?')}")
        else:
            log(f"    ✗ 失败 ({elapsed:.0f}s) — 继续下一夜")

        # 保存缓存
        cached["yasa_results"] = results
        with open(PIPELINE_CACHE, "wb") as f:
            pickle.dump(cached, f)

        # 强制 GC
        gc.collect()
        mem_now = get_memory_info()
        log(f"    内存: {mem_now['used_pct']:.1f}%")

    ok_count = sum(1 for r in results if r.get("yasa_ok"))
    log(f"\n  YASA 阶段完成: {ok_count}/{len(records)} 成功")

    return results


# ═══════════════════════════════════════════════════════════
#  阶段 2: 单通道特征提取 (轻量版)
# ═══════════════════════════════════════════════════════════

def run_feature_extraction(yasa_results, monitor):
    """对 YASA 成功的夜晚运行单通道特征提取 (子进程模式)。

    使用独立 Python 子进程运行 extract_single_night.py，
    避免 importlib 内联加载 101night_analy.py 时的阻塞。
    """
    import subprocess

    log("\n" + "=" * 60)
    log("阶段 2: 单通道特征提取 (轻量版, 子进程)")
    log("=" * 60)

    valid = [r for r in yasa_results if r.get("yasa_ok")]
    if not valid:
        log("  无 YASA 成功的夜晚，跳过特征提取")
        return []

    # 检查缓存
    if PIPELINE_CACHE.exists():
        with open(PIPELINE_CACHE, "rb") as f:
            cached = pickle.load(f)
        done_features = cached.get("feature_nights", set())
    else:
        cached = {}
        done_features = set()

    python_exe = sys.executable  # 用当前 Python 解释器
    extract_script = PROJECT_DIR / "extract_single_night.py"
    feature_results = []

    for r in valid:
        night = r["night"]

        if night in done_features:
            log(f"  N{night} — SKIP (已提取特征)")
            continue

        pct = get_memory_pct()
        if pct > MEMORY_THRESHOLD or monitor.alarm:
            log(f"  ⚠ 内存 {pct:.1f}% > {MEMORY_THRESHOLD}%，终止特征提取")
            break

        mff_path = find_mff_path(night)
        if not mff_path:
            log(f"  N{night} — 未找到 .mff 路径")
            continue

        log(f"\n  N{night}: 启动子进程特征提取...")
        t0 = time.time()

        cmd = [
            python_exe, str(extract_script),
            "--mff", str(mff_path),
            "--night", str(night),
            "--output", str(PROJECT_DIR),
        ]

        try:
            # 运行子进程，实时输出
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            )

            # 实时读取输出
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log(f"    [{night}] {line}")

            proc.wait(timeout=1800)  # 30 min timeout

            elapsed = time.time() - t0
            if proc.returncode == 0:
                log(f"    ✓ 完成 ({elapsed:.0f}s)")

                # 验证输出文件
                csv_path = PROJECT_DIR / f"features_night{night}_pipeline.csv"
                if csv_path.exists():
                    done_features.add(night)
                    cached["feature_nights"] = done_features
                    with open(PIPELINE_CACHE, "wb") as f:
                        pickle.dump(cached, f)
                    feature_results.append({
                        "night": night,
                        "csv_path": str(csv_path),
                    })
                else:
                    log(f"    ✗ CSV 未生成: {csv_path}")
            else:
                log(f"    ✗ 子进程退出码 {proc.returncode} ({elapsed:.0f}s)")

        except subprocess.TimeoutExpired:
            proc.kill()
            log(f"    ✗ 超时 (30min)")

        except Exception as e:
            log(f"    ✗ 异常: {e}")

        gc.collect()
        mem_now = get_memory_info()
        log(f"    内存: {mem_now['used_pct']:.1f}%")

    log(f"\n  特征提取完成: {len(feature_results)} 夜")
    return feature_results


def find_mff_path(night):
    """查找指定夜晚的 .mff 路径。"""
    # 先查本地
    local_matches = list(PROJECT_DIR.glob(f"Nathalie-{night}_*.mff"))
    if local_matches:
        return local_matches[0]

    # 再查数据目录
    data_matches = list(DATA_DIR.glob(f"Nathalie-{night}_*.mff"))
    if data_matches:
        return data_matches[0]

    return None


# ═══════════════════════════════════════════════════════════
#  阶段 3: 跨夜自相关
# ═══════════════════════════════════════════════════════════

def run_autocorr(yasa_results, monitor):
    """基于 YASA 结果运行跨夜自相关分析。

    注意: 3-5 夜数据太少，自相关可能不稳定，
    但仍运行以验证代码正确性。
    """
    log("\n" + "=" * 60)
    log("阶段 3: 跨夜自相关分析")
    log("=" * 60)

    valid = [r for r in yasa_results if r.get("yasa_ok")]
    if len(valid) < 3:
        log(f"  YASA 成功数 {len(valid)} < 3，跳过自相关 (样本太少)")
        return

    pct = get_memory_pct()
    if pct > MEMORY_THRESHOLD or monitor.alarm:
        log(f"  ⚠ 内存 {pct:.1f}% > {MEMORY_THRESHOLD}%，跳过")
        return

    # 写临时 batch_results.pkl 供 autocorr 脚本使用
    temp_results = PROJECT_DIR / "batch_results_temp.pkl"
    with open(temp_results, "wb") as f:
        pickle.dump(yasa_results, f)

    log(f"  临时结果: {temp_results} ({len(valid)} 夜)")

    # 导入 autocorr 脚本的函数
    sys.path.insert(0, str(PROJECT_DIR))
    from cross_night_autocorr import (
        load_nightly_metrics, compute_acf, compute_pacf,
        test_weekly_rhythm, plot_single_acf, plot_stage_autocorr_grid,
        plot_cross_correlation_matrix, plot_weekly_rhythm, plot_rolling_autocorr,
        generate_report,
    )

    # 替换 autocorr 脚本中的 CACHE_PATH
    import cross_night_autocorr as ac
    ac.CACHE_PATH = temp_results

    try:
        df = ac.load_nightly_metrics()
        log(f"  加载 {len(df)} 夜指标")

        if len(df) < 3:
            log("  夜数不足，跳过自相关")
            return

        log("  生成自相关图...")
        plot_single_acf(df["duration_h"], "duration",
                        "Sleep Duration (hours)", color="#2c7bb6")
        plot_single_acf(df["efficiency_pct"], "efficiency",
                        "Sleep Efficiency (%)", color="#e78ac3")
        plot_stage_autocorr_grid(df)

        log("  周周期检测...")
        for col, label in [("duration_h", "Duration"),
                            ("efficiency_pct", "Efficiency"),
                            ("n3_pct", "N3"), ("rem_pct", "REM")]:
            w = test_weekly_rhythm(df[col])
            log(f"    {label}: lag7 ACF={w['lag7_acf']:.3f}, "
                f"p={w['p_value']:.3f} {'***' if w['significant'] else ''}")

        generate_report(df)
        log("  自相关分析完成")

    except Exception as e:
        log(f"  自相关失败: {e}")


# ═══════════════════════════════════════════════════════════
#  阶段 4: 特征聚类
# ═══════════════════════════════════════════════════════════

def run_clustering(monitor):
    """对已提取的特征运行 QC + 聚类。

    使用 pipeline 生成的特征文件 (features_night*_pipeline.csv)。
    """
    log("\n" + "=" * 60)
    log("阶段 4: 特征 QC + 归一化 + 聚类 (k=5)")
    log("=" * 60)

    pct = get_memory_pct()
    if pct > MEMORY_THRESHOLD or monitor.alarm:
        log(f"  ⚠ 内存 {pct:.1f}% > {MEMORY_THRESHOLD}%，跳过")
        return

    # 查找 pipeline 特征文件
    feat_csvs = sorted(PROJECT_DIR.glob("features_night*_pipeline.csv"))
    if not feat_csvs:
        log("  未找到 pipeline 特征文件，跳过聚类")
        return

    log(f"  找到 {len(feat_csvs)} 个特征文件")

    # 导入聚类脚本
    sys.path.insert(0, str(PROJECT_DIR))

    # 加载并合并
    import pandas as pd
    dfs = []
    for fp in feat_csvs:
        try:
            df = pd.read_csv(fp)
            # 从文件名提取 night
            import re
            m = re.search(r'night(\d+)', fp.stem)
            if m and "night" not in df.columns:
                df["night"] = int(m.group(1))
            dfs.append(df)
            log(f"    {fp.name}: {len(df)} epochs, {len(df.columns)} cols")
        except Exception as e:
            log(f"    {fp.name}: 加载失败 — {e}")

    if len(dfs) < 2:
        log("  有效文件 < 2，跳过聚类")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    log(f"  合并: {len(df_all)} epochs, {len(df_all.columns)} columns")

    # 导入 QC + 聚类函数
    from feature_qc_cluster import (
        apply_qc, normalize_features, run_clustering as do_cluster,
        characterize_clusters, plot_tsne, plot_cluster_sizes,
        plot_cluster_profiles, plot_cluster_by_night, save_results,
        META_COLUMNS,
    )

    # QC
    log("  执行 QC...")
    df_clean, feature_cols, qc_summary = apply_qc(df_all)
    log(f"    QC 后: {len(df_clean)} epochs, {len(feature_cols)} features")

    if len(feature_cols) < 5:
        log("  特征数 < 5，跳过聚类")
        return

    # 归一化
    log("  归一化...")
    X_scaled, scaler, feature_cols = normalize_features(df_clean, feature_cols)

    # 聚类
    log("  聚类 k=5...")
    labels, model, cluster_metrics = do_cluster(X_scaled)
    profiles = characterize_clusters(X_scaled, labels, feature_cols, model)

    for k in sorted(profiles.keys()):
        log(f"    C{k}: {profiles[k]['size']} epochs ({profiles[k]['pct']:.0f}%)")

    # 可视化
    log("  生成可视化...")
    plot_tsne(X_scaled, labels, max_samples=min(5000, len(X_scaled)))
    plot_cluster_sizes(labels)
    plot_cluster_profiles(profiles, feature_cols)
    plot_cluster_by_night(df_clean, labels)

    # 保存
    save_results(df_clean, labels, model, profiles, feature_cols, qc_summary)

    log("  聚类分析完成")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="101night 流水线运行器")
    parser.add_argument("--nights", type=str,
                        default="31,40,41",
                        help="逗号分隔的夜晚编号 (默认: 31,40,41)")
    parser.add_argument("--skip-yasa", action="store_true",
                        help="跳过 YASA 分期")
    parser.add_argument("--skip-features", action="store_true",
                        help="跳过特征提取")
    parser.add_argument("--skip-autocorr", action="store_true",
                        help="跳过自相关")
    parser.add_argument("--skip-cluster", action="store_true",
                        help="跳过聚类")
    args = parser.parse_args()

    nights = [int(n.strip()) for n in args.nights.split(",")]

    log("=" * 60)
    log("101night Pipeline Runner")
    log(f"Target nights: {nights}")
    log(f"Memory threshold: {MEMORY_THRESHOLD}%")
    log("=" * 60)

    # 内存监控
    mem = get_memory_info()
    log(f"System: {mem['total_gb']:.1f} GB total, "
        f"{mem['available_gb']:.1f} GB available ({mem['used_pct']:.1f}%)")

    if mem['used_pct'] > MEMORY_THRESHOLD:
        log(f"⚠ 启动时内存已超阈值 ({mem['used_pct']:.1f}% > {MEMORY_THRESHOLD}%)!")
        log("请关闭其他进程后重试。")
        return

    monitor = MemoryMonitor(threshold=MEMORY_THRESHOLD)
    monitor.start()

    try:
        # 阶段 1: YASA
        if not args.skip_yasa:
            yasa_results = run_yasa_batch(nights, monitor)
        else:
            log("跳过 YASA 分期")
            if PIPELINE_CACHE.exists():
                with open(PIPELINE_CACHE, "rb") as f:
                    cached = pickle.load(f)
                yasa_results = cached.get("yasa_results", [])
            else:
                yasa_results = []

        if monitor.alarm:
            log("⚠ 内存超阈值，终止流水线")
            return

        # 阶段 2: 特征提取
        if not args.skip_features:
            feat_results = run_feature_extraction(yasa_results, monitor)
        else:
            log("跳过特征提取")
            feat_results = []

        if monitor.alarm:
            log("⚠ 内存超阈值，终止流水线")
            return

        # 阶段 3: 自相关
        if not args.skip_autocorr:
            run_autocorr(yasa_results, monitor)
        else:
            log("跳过自相关分析")

        if monitor.alarm:
            log("⚠ 内存超阈值，终止流水线")
            return

        # 阶段 4: 聚类
        if not args.skip_cluster:
            run_clustering(monitor)
        else:
            log("跳过聚类分析")

    except KeyboardInterrupt:
        log("\n⚠ 用户中断")
    except Exception as e:
        log(f"\n⚠ 流水线异常: {e}")
        import traceback
        log(traceback.format_exc())
    finally:
        monitor.stop()
        peak = monitor.peak_pct
        log(f"\n内存峰值: {peak:.1f}%")
        if peak > MEMORY_THRESHOLD:
            log(f"⚠ 超过 {MEMORY_THRESHOLD}% 阈值，已触发保护!")

    log("\n" + "=" * 60)
    log("流水线完成")
    log(f"日志: {LOG_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    main()
