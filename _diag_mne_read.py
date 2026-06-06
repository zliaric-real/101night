# -*- coding: utf-8 -*-
"""_diag_mne_read.py — 复现: import yasa 先, 再 MNE (多次)"""
import time, gc
import numpy as np
import yasa
import mne

MFF = "I:/101Night/Nathalie-40_20171011_121248.mff"

def test_mne_read(label):
    t0 = time.time()
    raw = mne.io.read_raw_egi(MFF, preload=False, verbose=False)
    raw.pick(["E21", "E67"])
    raw.load_data()
    data = raw.get_data(units='uV')
    dt = time.time() - t0
    del raw, data; gc.collect()
    print(f"  {label}: {dt:.1f}s, shape={data.shape if 'data' in dir() else 'N/A'}")

for i in range(5):
    test_mne_read(f"call {i+1}")
print("\nDone.")
