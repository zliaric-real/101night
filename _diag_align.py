# -*- coding: utf-8 -*-
"""全列相关扫描: MNE E21 vs signal1.bin 257列的每一列"""
import numpy as np
import mne
from pathlib import Path

MFF = "I:/101Night/Nathalie-40_20171011_121248.mff"
EEG_CH = "E21"
REAL_NCH = 257
START = 100000  # 跳过开头校准区
N_SAMPLES = 5000

# MNE E21
raw = mne.io.read_raw_egi(MFF, preload=False, verbose=False)
raw.pick([EEG_CH])
raw.load_data()
data_mne = raw.get_data(units='uV')[0][START:START+N_SAMPLES]
print(f"MNE E21[{START}:{START+N_SAMPLES}]: mean={data_mne.mean():.1f}, std={data_mne.std():.1f}")
print(f"  values: min={data_mne.min():.0f}, max={data_mne.max():.0f}")

# signal1.bin: read N_SAMPLES * REAL_NCH values starting from START
signal_path = Path(MFF) / "signal1.bin"
with open(str(signal_path), "rb") as f:
    f.seek(START * REAL_NCH * 4)
    raw_bytes = f.read(N_SAMPLES * REAL_NCH * 4)
all_raw = np.frombuffer(raw_bytes, dtype=np.float32).reshape(N_SAMPLES, REAL_NCH).T  # (257, N_SAMPLES)

# Correlate each column with MNE E21
best_corr, best_col = -1, -1
for col in range(REAL_NCH):
    corr = np.corrcoef(all_raw[col], data_mne)[0, 1]
    if corr > best_corr:
        best_corr, best_col = corr, col

print(f"\nBest match: col {best_col}, corr={best_corr:.6f}")
print(f"  raw col {best_col}: mean={all_raw[best_col].mean():.0f}, std={all_raw[best_col].std():.0f}")
print(f"  MNE E21:           mean={data_mne.mean():.0f}, std={data_mne.std():.0f}")

# Also print top 5 matches
corrs = [(np.corrcoef(all_raw[c], data_mne)[0,1], c) for c in range(REAL_NCH)]
corrs.sort(reverse=True)
print(f"\nTop 5 correlations:")
for corr, col in corrs[:5]:
    print(f"  col {col}: corr={corr:.6f}, mean={all_raw[col].mean():.0f}, std={all_raw[col].std():.0f}")

# If best correlation is low, check if any column has similar statistics
print(f"\nColumns with similar stats to MNE E21:")
for col in range(REAL_NCH):
    if abs(all_raw[col].std() - data_mne.std()) < 50:
        print(f"  col {col}: mean={all_raw[col].mean():.0f}, std={all_raw[col].std():.0f}")
