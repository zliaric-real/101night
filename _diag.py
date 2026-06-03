"""诊断脚本：逐步计时，定位 chunked read 瓶颈"""
import sys, os, time
from pathlib import Path

MFF_DIR = r"I:\101Night\Nathalie-40_20171011_121248.mff"

def t(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

T0 = time.time()
t("start")

# 1. Python imports
t("import numpy...")
import numpy as np
t(f"numpy {np.__version__} ok")

t("import mne...")
import mne
t(f"mne {mne.__version__} ok")

# 2. MNE metadata
t("MNE read_raw_egi(preload=False)...")
raw = mne.io.read_raw_egi(MFF_DIR, preload=False, verbose=False)
t(f"done: {len(raw.ch_names)}ch, {raw.info['sfreq']}Hz, {raw.n_times} samples, {raw.n_times/raw.info['sfreq']/3600:.1f}h")

ch_map = {ch: i for i, ch in enumerate(raw.ch_names)}
eeg_idx = ch_map["E21"]
nch = len(raw.ch_names)
t(f"E21 idx={eeg_idx}, n_channels={nch}")

# 3. File open
sp = Path(MFF_DIR) / "signal1.bin"
fsize = sp.stat().st_size
t(f"signal1.bin: {fsize/1e9:.2f} GB, exists={sp.exists()}")

t("opening file...")
f = open(str(sp), "rb")
t("file opened")

# 4. First chunk
CHUNK = 2_000_000
t(f"reading first chunk ({CHUNK} samples)...")
c = np.fromfile(f, dtype="float32", count=CHUNK)
t(f"chunk: {len(c)} values, {c.nbytes/1e6:.1f} MB")

# 5. Reshape + extract
ns = len(c) // nch
t(f"reshape to ({ns}, {nch})...")
c2d = c[:ns * nch].reshape(ns, nch)
eeg0 = c2d[:, eeg_idx]
t(f"E21 data: shape={eeg0.shape}, min={eeg0.min():.6f}, max={eeg0.max():.6f}")

f.close()

# 6. Now test full read with small file or with timer
t("=== Full chunked read (just E21, no concat) ===")
f2 = open(str(sp), "rb")
chunk_count = 0
bytes_total = 0
while True:
    c = np.fromfile(f2, dtype="float32", count=CHUNK)
    if len(c) == 0:
        break
    ns = len(c) // nch
    if ns == 0:
        break
    c2d = c[:ns * nch].reshape(ns, nch)
    _ = c2d[:, eeg_idx]  # just extract, don't store
    chunk_count += 1
    bytes_total += c.nbytes
    if chunk_count % 100 == 0:
        t(f"  {chunk_count} chunks, {bytes_total/1e9:.2f} GB read")

f2.close()
t(f"FULL READ DONE: {chunk_count} chunks, {bytes_total/1e9:.2f} GB total")
t("=== DIAGNOSTIC PASSED ===")
