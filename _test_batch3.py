"""Quick batch test — first 3 nights only"""
import sys, os
sys.path.insert(0, "E:/idea/101night")
import importlib.util
spec = importlib.util.spec_from_file_location("analy", "E:/idea/101night/101night_analy.py")
analy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analy)
SleepEEGFeatureExtractor = analy.SleepEEGFeatureExtractor

from pathlib import Path
import re
from datetime import datetime

DATA_DIR = Path("I:/101Night")
mffs = sorted(DATA_DIR.glob("Nathalie-*.mff"))
records = []
for mff_path in mffs:
    name = mff_path.name
    match = re.match(r"Nathalie-(\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.mff", name)
    if match:
        night = int(match.group(1))
        y, m, d = int(match.group(2)), int(match.group(3)), int(match.group(4))
        hh, mm, ss = int(match.group(5)), int(match.group(6)), int(match.group(7))
        records.append({"night": night, "date": f"{y}-{m:02d}-{d:02d}",
                        "datetime": datetime(y,m,d,hh,mm,ss), "path": str(mff_path)})
records.sort(key=lambda r: r["datetime"])

# Only first 3
for i, rec in enumerate(records[:3]):
    night = rec["night"]
    print(f"\n[{i+1}/3] Night {night}  {rec['date']}", flush=True)
    try:
        ext = SleepEEGFeatureExtractor(rec["path"], eeg_channel="E21")
        # YASA
        print(f"  YASA...", flush=True)
        stages = ext.sleep_stages_yasa()
        if stages is not None:
            ss = ext.features.get("sleep_stages", {})
            print(f"  YASA OK: {ss.get('stage_pct', 'N/A')}", flush=True)
        # SSSM
        print(f"  SSSM...", flush=True)
        waves = ext.feature_waves()
        if waves is not None:
            unique, counts = __import__('numpy').unique(waves, return_counts=True)
            labels = {0:"BG",1:"Spindle",2:"SW",3:"KC",4:"Saw",5:"Vx",6:"Ar"}
            info = {labels.get(u,u):c for u,c in zip(unique, counts)}
            print(f"  SSSM OK: {info}", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\n=== DONE ===", flush=True)
