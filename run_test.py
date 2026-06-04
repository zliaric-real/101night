# -*- coding: utf-8 -*-
"""Wrapper — redirects all output to a log file for real-time monitoring."""
import sys, os, time, subprocess

LOG = r"E:\idea\101night\test_night40_log.txt"
PYTHON = r"C:\Users\PC\anaconda3\envs\eeg_101night\python.exe"
SCRIPT = r"E:\idea\101night\extract_single_night.py"
MFF = r"E:\idea\101night\Nathalie-40_20171011_121248.mff"

# Accept extra args like --skip-yasa --skip-source
extra_args = sys.argv[1:]

with open(LOG, "w", encoding="utf-8") as f:
    f.write(f"=== Night 40 Test Start: {time.ctime()} ===\n")
    f.write(f"=== Extra args: {extra_args} ===\n")
    f.flush()

cmd = [PYTHON, "-u", SCRIPT, "--mff", MFF, "--night", "40"] + extra_args
proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd=r"E:\idea\101night", text=True, bufsize=1,
)

with open(LOG, "a", encoding="utf-8") as f:
    for line in proc.stdout:
        f.write(line)
        f.flush()

proc.wait()
with open(LOG, "a", encoding="utf-8") as f:
    f.write(f"\n=== Exit code: {proc.returncode} ===\n")
    f.write(f"=== Night 40 Test End: {time.ctime()} ===\n")
