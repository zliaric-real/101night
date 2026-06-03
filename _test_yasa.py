"""Minimal single-night YASA test — verify end-to-end pipeline"""
import time, sys
from pathlib import Path
import numpy as np

MFF = r"I:\101Night\Nathalie-40_20171011_121248.mff"  # 3.3h, smaller = faster test
EEG_CH, EOG_L, EOG_R = "E21", "E67", "E219"
CHUNK = 2_000_000

def t(msg):
    print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)

T0 = time.time()
t("start")

# 1. Metadata via MNE
import mne
raw = mne.io.read_raw_egi(MFF, preload=False, verbose=False)
ch_names = raw.ch_names
nch = len(ch_names)
sfreq = raw.info["sfreq"]
nt = raw.n_times
t(f"MNE: {nch}ch, {sfreq}Hz, {nt} samples = {nt/sfreq/3600:.1f}h")

ch_map = {ch: i for i, ch in enumerate(ch_names)}
eeg_idx = ch_map[EEG_CH]
eogL_idx = ch_map[EOG_L]
eogR_idx = ch_map[EOG_R]
t(f"E21={eeg_idx}, E67={eogL_idx}, E219={eogR_idx}")

# 2. Chunked read with .copy()
sp = Path(MFF) / "signal1.bin"
t(f"reading {sp.stat().st_size/1e9:.2f} GB...")
eeg_p, eogL_p, eogR_p = [], [], []
with open(str(sp), "rb") as f:
    while True:
        c = np.fromfile(f, dtype="float32", count=CHUNK)
        if len(c) == 0: break
        ns = len(c) // nch
        if ns == 0: break
        c2d = c[:ns*nch].reshape(ns, nch)
        eeg_p.append(c2d[:, eeg_idx].copy())
        eogL_p.append(c2d[:, eogL_idx].copy())
        eogR_p.append(c2d[:, eogR_idx].copy())

eeg = np.concatenate(eeg_p).astype("float64") * 1e6
eogL = np.concatenate(eogL_p).astype("float64") * 1e6
eogR = np.concatenate(eogR_p).astype("float64") * 1e6
eog = eogL - eogR
t(f"data loaded: EEG {eeg.nbytes/1e6:.0f}MB, EOG bipolar")

# 3. Build RawArray for YASA
t("building RawArray...")
combined = np.vstack([eeg, eog])
info = mne.create_info([EEG_CH, "EOG"], sfreq, ch_types=["eeg", "eog"])
raw_stage = mne.io.RawArray(combined, info, verbose=False)
t(f"RawArray ready: {raw_stage.n_times/sfreq/3600:.1f}h")

# 4. YASA
t("YASA SleepStaging...")
import yasa
sls = yasa.SleepStaging(raw_stage, eeg_name=EEG_CH, eog_name="EOG")
t("YASA predict...")
hypno = sls.predict()
t(f"YASA done: {len(hypno)} epochs")

labels = {0:"Wake",1:"N1",2:"N2",3:"N3",4:"REM"}
counts = {labels.get(s,s): int((hypno==s).sum()) for s in np.unique(hypno)}
total = len(hypno)
pct = {k: f"{v/total*100:.1f}%" for k,v in counts.items()}
t(f"stages: {pct}")
t("=== ALL OK ===")
