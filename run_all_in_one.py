# -*- coding: utf-8 -*-
"""
run_all_in_one.py — 单进程全流程 (避免重复下载 YASA 模型)
=========================================================
在一个 Python 进程中依次执行: YASA 分期 → 特征提取 → 自相关 → 聚类。
全程只导入一次 yasa / TF，避免模型重复下载。

用法: python run_all_in_one.py --nights 40,41
"""

import os, sys, re, gc, time, pickle, threading
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("I:/101Night")
LOG_PATH = PROJECT_DIR / "allinone_log.txt"

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush()

# ── 内存监控 ──
def mem_pct():
    try:
        import psutil; return psutil.virtual_memory().percent
    except: return -1

MEM_LIMIT = 80.0

# ═══════════════════════════════════════════════════════════
log("="*60)
log("ALL-IN-ONE Pipeline (single process, single YASA import)")
log(f"Mem limit: {MEM_LIMIT}%")
log("="*60)

pct = mem_pct()
log(f"Start mem: {pct:.1f}%")

# ═══ STEP 0: Import everything ONCE ═══
log("Importing YASA (one-time model download, may take 5-10 min)...")
t0 = time.time()
import yasa
import mne
import numpy as np
import pandas as pd
log(f"YASA imported ({time.time()-t0:.0f}s)")

# ═══ STEP 1: Scan nights ═══
sys.path.insert(0, str(PROJECT_DIR))
from batch_analyze import scan_mff_dirs, process_one_night

all_records = scan_mff_dirs()
target = {int(n.strip()) for n in sys.argv[-1].split(",")} if len(sys.argv) > 1 else {40}
records = [r for r in all_records if r["night"] in target]
log(f"Nights: {[r['night'] for r in records]}")

# ═══ STEP 2: YASA staging ═══
log("\n--- YASA Staging ---")
yasa_results = []
for rec in records:
    pct = mem_pct()
    if pct > MEM_LIMIT:
        log(f"MEM {pct:.1f}% > {MEM_LIMIT}% — STOP")
        break
    log(f"N{rec['night']} processing...")
    t0 = time.time()
    result = process_one_night(rec)
    elapsed = time.time() - t0
    ok = result.get("yasa_ok")
    log(f"  {'OK' if ok else 'FAIL'} ({elapsed:.0f}s) {result.get('stage_pct','')}")
    yasa_results.append(result)
    gc.collect()

ok_nights = [r for r in yasa_results if r.get("yasa_ok")]
log(f"YASA done: {len(ok_nights)}/{len(records)} OK")

if not ok_nights:
    log("No YASA success — stopping")
    sys.exit(1)

# Save YASA results
with open(PROJECT_DIR / "batch_results.pkl", "wb") as f:
    pickle.dump(yasa_results, f)

# ═══ STEP 3: Feature extraction (light) ═══
log("\n--- Feature Extraction ---")
import importlib.util
_analy_spec = importlib.util.spec_from_file_location("analy", str(PROJECT_DIR / "101night_analy.py"))
_analy = importlib.util.module_from_spec(_analy_spec)
_analy_spec.loader.exec_module(_analy)
SleepEEGFeatureExtractor = _analy.SleepEEGFeatureExtractor

light_groups = ('basic', 'time_domain', 'frequency', 'entropy', 'complexity')

for r in ok_nights:
    night = r["night"]
    pct = mem_pct()
    if pct > MEM_LIMIT:
        log(f"MEM {pct:.1f}% — STOP")
        break

    mff_path = None
    for rec in records:
        if rec["night"] == night:
            mff_path = rec["path"]; break
    if not mff_path:
        log(f"N{night}: no path"); continue

    log(f"N{night}: feature extraction...")
    t0 = time.time()
    try:
        ext = SleepEEGFeatureExtractor(mff_path, eeg_channel='E21',
                                        load_all_channels=False)
        ext.run_all(epoch_sec=30, groups=light_groups, skip_yasa=True, skip_sssm=True)
        df = ext.to_dataframe(epoch_sec=30)
        csv_path = PROJECT_DIR / f"features_night{night}_pipeline.csv"
        df.to_csv(csv_path, index=False)
        elapsed = time.time() - t0
        log(f"  OK {len(df)} epochs, {len(df.columns)} features ({elapsed:.0f}s)")
        del ext, df; gc.collect()
    except Exception as e:
        log(f"  FAIL: {e}")

# ═══ STEP 4: Autocorrelation ═══
log("\n--- Autocorrelation ---")
if len(ok_nights) >= 3:
    try:
        import cross_night_autocorr as ac
        ac.CACHE_PATH = PROJECT_DIR / "batch_results.pkl"
        df = ac.load_nightly_metrics()
        ac.plot_single_acf(df["duration_h"], "duration", "Duration")
        ac.plot_single_acf(df["efficiency_pct"], "efficiency", "Efficiency")
        ac.plot_stage_autocorr_grid(df)
        ac.generate_report(df)
        log("Autocorr done")
    except Exception as e:
        log(f"Autocorr FAIL: {e}")
else:
    log(f"Skip autocorr: need >=3 nights, got {len(ok_nights)}")

# ═══ STEP 5: Clustering ═══
log("\n--- Clustering ---")
feat_files = sorted(PROJECT_DIR.glob("features_night*_pipeline.csv"))
if len(feat_files) >= 2:
    try:
        from feature_qc_cluster import (apply_qc, normalize_features, run_clustering,
                                         characterize_clusters, plot_tsne, plot_cluster_sizes,
                                         plot_cluster_profiles, plot_cluster_by_night,
                                         save_results, META_COLUMNS)
        dfs = []
        for fp in feat_files:
            df = pd.read_csv(fp)
            m = re.search(r'night(\d+)', fp.stem)
            if m and 'night' not in df.columns:
                df['night'] = int(m.group(1))
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
        log(f"Loaded {len(df_all)} epochs, {len(df_all.columns)} cols")

        df_clean, fcols, qc = apply_qc(df_all)
        log(f"QC: {len(df_clean)} epochs, {len(fcols)} features")

        X, scaler, fcols = normalize_features(df_clean, fcols)
        labels, model, metrics = run_clustering(X)
        profiles = characterize_clusters(X, labels, fcols, model)

        for k in sorted(profiles):
            log(f"  C{k}: {profiles[k]['size']} ({profiles[k]['pct']:.0f}%)")

        plot_tsne(X, labels, min(5000, len(X)))
        plot_cluster_sizes(labels)
        plot_cluster_profiles(profiles, fcols)
        plot_cluster_by_night(df_clean, labels)
        save_results(df_clean, labels, model, profiles, fcols, qc)
        log("Clustering done")
    except Exception as e:
        log(f"Clustering FAIL: {e}")
        import traceback; traceback.print_exc()
else:
    log(f"Skip clustering: need >=2 CSVs, got {len(feat_files)}")

# ═══ DONE ═══
pct = mem_pct()
log(f"\nALL DONE. Final mem: {pct:.1f}%")
log(f"Log: {LOG_PATH}")
