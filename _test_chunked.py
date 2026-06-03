"""Minimal test: chunked read of a single night's signal1.bin"""
import sys, os, time
from pathlib import Path
import numpy as np

MFF_PATH = r"I:\101Night\Nathalie-40_20171011_121248.mff"
EEG_CH = "E21"

print("=== Chunked read test ===", flush=True)

# Read metadata via MNE
print("Loading MNE...", flush=True)
import mne
t0 = time.time()
raw = mne.io.read_raw_egi(MFF_PATH, preload=False, verbose=False)
print(f"  MNE metadata: {time.time()-t0:.1f}s", flush=True)
print(f"  Channels: {len(raw.ch_names)}, sfreq: {raw.info['sfreq']}, n_times: {raw.n_times}", flush=True)
print(f"  Duration: {raw.n_times/raw.info['sfreq']/3600:.1f}h", flush=True)

ch_map = {ch: i for i, ch in enumerate(raw.ch_names)}
eeg_idx = ch_map[EEG_CH]
have_eog = 'E67' in ch_map and 'E219' in ch_map
eogL_idx = ch_map.get('E67')
eogR_idx = ch_map.get('E219')
n_channels = len(raw.ch_names)
print(f"  E21 idx={eeg_idx}, EOG: {have_eog}, n_channels={n_channels}", flush=True)

# Chunked read
signal_path = Path(MFF_PATH) / "signal1.bin"
file_size = signal_path.stat().st_size
print(f"  signal1.bin: {file_size/1e9:.2f} GB", flush=True)

CHUNK = 2_000_000
eeg_parts = []
n_chunks = 0
t0 = time.time()

with open(str(signal_path), "rb") as f:
    while True:
        chunk = np.fromfile(f, dtype="float32", count=CHUNK)
        if len(chunk) == 0:
            break
        n_samples = len(chunk) // n_channels
        if n_samples == 0:
            break
        chunk_2d = chunk[:n_samples * n_channels].reshape(n_samples, n_channels)
        eeg_parts.append(chunk_2d[:, eeg_idx].copy())
        n_chunks += 1
        if n_chunks % 50 == 0:
            print(f"  ... {n_chunks} chunks, {time.time()-t0:.1f}s", flush=True)

eeg = np.concatenate(eeg_parts)
print(f"  Done: {n_chunks} chunks in {time.time()-t0:.1f}s", flush=True)
print(f"  EEG data: {eeg.shape}, dtype={eeg.dtype}, min={eeg.min():.4f}, max={eeg.max():.4f}", flush=True)

# Convert to μV
eeg_uv = eeg.astype("float64") * 1e6
print(f"  μV: min={eeg_uv.min():.1f}, max={eeg_uv.max():.1f}, mean={eeg_uv.mean():.1f}", flush=True)
print(f"  Memory: {eeg_uv.nbytes/1e6:.1f} MB", flush=True)

print("\n=== TEST PASSED ===", flush=True)
