"""Test chunked I/O for all 34 nights (NO YASA) — verify pipeline only"""
import time, re, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import mne

DATA = Path("I:/101Night")
EEG_CH, EOG_L, EOG_R = "E21", "E67", "E219"
CHUNK = 2_000_000
T0 = time.time()

# Scan & sort
records = []
for mff in sorted(DATA.glob("Nathalie-*.mff")):
    m = re.match(r"Nathalie-(\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.mff", mff.name)
    if m:
        night = int(m.group(1))
        dt = datetime(int(m.group(2)),int(m.group(3)),int(m.group(4)),int(m.group(5)),int(m.group(6)),int(m.group(7)))
        records.append({"night": night, "date": f"{dt.year}-{dt.month:02d}-{dt.day:02d}", "path": str(mff), "dt": dt})

records.sort(key=lambda r: r["dt"])
print(f"Found {len(records)} nights", flush=True)

ok = 0
fail_mne = 0
fail_xml = 0
fail_io = 0

for i, rec in enumerate(records):
    night = rec["night"]
    label = f"[{i+1}/{len(records)}] N{night}"

    # MNE metadata
    try:
        raw = mne.io.read_raw_egi(rec["path"], preload=False, verbose=False)
        ch_names = raw.ch_names
        nch = len(ch_names)
        sfreq = raw.info["sfreq"]
        nt = raw.n_times
    except Exception as e:
        print(f"  {label}: MNE FAILED — {e}", flush=True)
        fail_mne += 1
        continue  # skip for now — XML fallback not tested

    ch_map = {ch: i for i, ch in enumerate(ch_names)}
    if EEG_CH not in ch_map:
        print(f"  {label}: E21 not found", flush=True)
        continue
    eeg_idx = ch_map[EEG_CH]
    eogL_idx = ch_map.get(EOG_L)
    eogR_idx = ch_map.get(EOG_R)

    # Chunked read
    try:
        sp = Path(rec["path"]) / "signal1.bin"
        eeg_p, eogL_p, eogR_p = [], [], []
        with open(str(sp), "rb") as f:
            while True:
                c = np.fromfile(f, dtype="float32", count=CHUNK)
                if len(c) == 0: break
                ns = len(c) // nch
                if ns == 0: break
                c2d = c[:ns*nch].reshape(ns, nch)
                eeg_p.append(c2d[:, eeg_idx].copy())
                if eogL_idx is not None:
                    eogL_p.append(c2d[:, eogL_idx].copy())
                    eogR_p.append(c2d[:, eogR_idx].copy())

        eeg = np.concatenate(eeg_p).astype("float64") * 1e6
        dur = nt / sfreq / 3600
        dur_read = nt / eeg.shape[0] * dur if eeg.shape[0] != nt else dur
        msg = f"  {label}: {nch}ch, {dur:.1f}h, E21={eeg.shape[0]}samp, {eeg.nbytes/1e6:.0f}MB, OK"
        ok += 1
    except Exception as e:
        msg = f"  {label}: IO FAILED — {e}"
        fail_io += 1

    print(msg, flush=True)

elapsed = time.time() - T0
print(f"\n=== Results: {ok} OK, {fail_mne} MNE fail, {fail_io} IO fail ===", flush=True)
print(f"Time: {elapsed:.1f}s ({elapsed/34:.1f}s/night)", flush=True)
