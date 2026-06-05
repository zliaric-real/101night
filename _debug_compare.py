"""Compare MNE vs chunked raw values for E21 — diagnose gain factor."""
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import mne

MFF = r"I:\101Night\Nathalie-40_20171011_121248.mff"
EEG_CH = 'E21'
CHUNK_VALUES = 500_000

# ── 1. Parse sensorLayout.xml for channel names ──
mff = Path(MFF)
layout_path = mff / "sensorLayout.xml"
ltree = ET.parse(str(layout_path))
lroot = ltree.getroot()
lns = lroot.tag.split('}')[0] + '}' if '}' in lroot.tag else ''
ch_names = []
for sensor in lroot.find(f'{lns}sensors').findall(f'{lns}sensor'):
    stype = int(sensor.find(f'{lns}type').text)
    snum = int(sensor.find(f'{lns}number').text)
    sname = sensor.find(f'{lns}name')
    name = sname.text.strip() if sname is not None and sname.text else ""
    if not name and stype == 0:
        name = f"E{snum}"
    ch_names.append(name)
eeg_idx = ch_names.index(EEG_CH)

# ── 2. Read signal1.bin raw float32 (NO unit conversion) ──
signal_path = mff / "signal1.bin"
n_channels = 260  # EGI 256

# Read first 100,000 samples of E21 only
buf = np.empty(CHUNK_VALUES, dtype=np.float32)
buf_mv = memoryview(buf).cast('B')
chunks = []

with open(str(signal_path), "rb") as f:
    while True:
        n_read = f.readinto(buf_mv)
        if n_read == 0:
            break
        n_values = n_read // 4
        if n_values < n_channels:
            break
        n_samples = n_values // n_channels
        if n_samples == 0:
            break
        chunk_2d = buf[:n_samples * n_channels].reshape(n_samples, n_channels)
        chunks.append(chunk_2d[:, eeg_idx].copy())

raw_float32 = np.concatenate(chunks)  # raw values from file
print(f"=== Raw float32 from signal1.bin ===")
print(f"  shape: {raw_float32.shape}, dtype: {raw_float32.dtype}")
print(f"  min: {raw_float32.min():.10f}, max: {raw_float32.max():.10f}")
print(f"  mean: {raw_float32.mean():.10f}, std: {raw_float32.std():.10f}")
print(f"  前10值: {raw_float32[:10]}")

# ── 3. MNE (μV) ──
raw_mne = mne.io.read_raw_egi(str(MFF), preload=False, verbose=False)
raw_mne.pick([EEG_CH])
raw_mne.load_data()
mne_uV = raw_mne.get_data(units='uV')[0].astype(np.float64)
print(f"\n=== MNE μV ===")
print(f"  shape: {mne_uV.shape}")
print(f"  min: {mne_uV.min():.2f}, max: {mne_uV.max():.2f}")
print(f"  mean: {mne_uV.mean():.2f}, std: {mne_uV.std():.2f}")
print(f"  前10值: {mne_uV[:10]}")

# ── 4. Compare ──
min_len = min(len(raw_float32), len(mne_uV))
r = raw_float32.astype(np.float64)[:min_len]
m = mne_uV[:min_len]

# raw → μV: what multiplier?
ratio = m / r
print(f"\n=== MNE(μV) / raw_float32 ===")
print(f"  ratio mean: {ratio.mean():.10f}, median: {np.median(ratio):.10f}")
print(f"  ratio unique: {len(np.unique(ratio.round(10)))}")
if len(np.unique(ratio.round(10))) < 10:
    for v in sorted(np.unique(ratio.round(10))):
        print(f"  {v}")

# Check if it's a clean power-of-10 or standard gain
print(f"\n  Ratio / 1e6 = {ratio.mean()/1e6:.10f}")
print(f"  Ratio / 1e9 = {ratio.mean()/1e9:.10f}")
print(f"  1 / Ratio = {1/ratio.mean():.10e}")
