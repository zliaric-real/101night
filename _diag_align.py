# -*- coding: utf-8 -*-
"""MNE vs raw: 检查是否有DC偏移/缩放/标定"""
import numpy as np
import mne
from pathlib import Path
import xml.etree.ElementTree as ET

MFF = "I:/101Night/Nathalie-40_20171011_121248.mff"
EEG_CH = "E21"
REAL_NCH = 257

raw = mne.io.read_raw_egi(MFF, preload=False, verbose=False)
raw.pick([EEG_CH])
raw.load_data()
data_mne = raw.get_data(units='uV')[0]

signal_path = Path(MFF) / "signal1.bin"

# Read raw E21 column (nch=257, col 20) for first 1000 samples
raw_vals = []
with open(str(signal_path), "rb") as f:
    for i in range(1000):
        f.seek(i * REAL_NCH * 4 + 20 * 4)
        raw_vals.append(np.frombuffer(f.read(4), dtype=np.float32)[0])
raw_vals = np.array(raw_vals)

# Compare
print(f"Raw col20: mean={raw_vals.mean():.1f}, std={raw_vals.std():.1f}, first10={raw_vals[:10]}")
print(f"MNE  E21:  mean={data_mne[:1000].mean():.1f}, std={data_mne[:1000].std():.1f}, first10={data_mne[:10]}")

# Check: is it a simple offset or scaling?
diff = data_mne[:1000] - raw_vals
print(f"\nDiff: mean={diff.mean():.1f}, std={diff.std():.1f}")
ratio = data_mne[:1000] / (raw_vals + 1e-12)
print(f"Ratio: mean={ratio.mean():.3f}")

# Check specific comparison
for off in [0, 100, 500, 999]:
    with open(str(signal_path), "rb") as f:
        f.seek(off * REAL_NCH * 4 + 20 * 4)
        rv = np.frombuffer(f.read(4), dtype=np.float32)[0]
    print(f"  sample {off:5d}: MNE={data_mne[off]:10.1f}  raw={rv:10.1f}  diff={data_mne[off]-rv:10.1f}")

# Check info1.xml for calibration
info1 = Path(MFF) / "info1.xml"
if info1.exists():
    itree = ET.parse(str(info1))
    iroot = itree.getroot()
    ins = iroot.tag.split('}')[0] + '}' if '}' in iroot.tag else ''
    for cal in iroot.iter(f'{ins}calibrations'):
        cal_text = cal.text.strip() if cal.text else ""
        if cal_text:
            print(f"\ncalibrations: {cal_text[:200]}")
        else:
            print(f"\ncalibrations: EMPTY")
    for sr in iroot.iter(f'{ins}sampRate'):
        print(f"sampRate: {sr.text}")
