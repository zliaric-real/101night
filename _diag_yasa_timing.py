# -*- coding: utf-8 -*-
"""_diag_yasa_timing.py — 精确计时, 每步 flush"""
import time, gc, sys
import numpy as np
import mne

MFF = "I:/101Night/Nathalie-40_20171011_121248.mff"
EEG_CH, EOG_CH = "E21", "E67"

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("YASA 计时诊断 — Night 40 (强制 flush)")
log("=" * 60)

# 先做 3 次纯 MNE 读, 确认稳定性
for i in range(3):
    t0 = time.time()
    raw = mne.io.read_raw_egi(MFF, preload=False, verbose=False)
    raw.pick([EEG_CH, EOG_CH])
    raw.load_data()
    data = raw.get_data(units='uV')
    dt = time.time() - t0
    del raw, data; gc.collect()
    log(f"[pre-check {i+1}] MNE read: {dt:.1f}s")

# 现在 import yasa 后再测
log("\n[import] importing yasa...")
import yasa
log(f"[import] done")

log("\n[MNE+EOG] MNE read after yasa import...")
t0 = time.time()
raw = mne.io.read_raw_egi(MFF, preload=False, verbose=False)
log(f"  read_raw_egi: {time.time()-t0:.1f}s")
raw.pick([EEG_CH, EOG_CH])
raw.load_data()
log(f"  load_data: {time.time()-t0:.1f}s")
data = raw.get_data(units='uV')
sfreq = raw.info['sfreq']
log(f"  get_data: {time.time()-t0:.1f}s, shape={data.shape}")

# 构建 RawArray
log("\n[RawArray] building...")
info = mne.create_info([EEG_CH, EOG_CH], sfreq, ch_types=['eeg', 'eog'])
raw_stage = mne.io.RawArray(data, info, verbose=False)
del raw, data; gc.collect()
log(f"[RawArray] done")

# YASA
log("\n[YASA init] SleepStaging...")
t0 = time.time()
sls = yasa.SleepStaging(raw_stage, eeg_name=EEG_CH, eog_name=EOG_CH)
log(f"[YASA init] {time.time()-t0:.1f}s")

log("\n[YASA predict] calling predict()...")
t0 = time.time()
hypno = np.asarray(sls.predict(), dtype=int)
log(f"[YASA predict] {time.time()-t0:.1f}s")

total = len(hypno)
labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
counts = {labels.get(s, s): int((hypno == s).sum()) for s in np.unique(hypno)}

log(f"\n{'='*60}")
log(f"Epochs: {total} | { {k:round(v/total*100,1) for k,v in counts.items()} }")
log("=" * 60)
