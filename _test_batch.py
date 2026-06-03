"""minimal test"""
import sys, os
print("=== START ===", flush=True)

sys.path.insert(0, "E:/idea/101night")
print("path inserted", flush=True)

import importlib.util
spec = importlib.util.spec_from_file_location("analy", "E:/idea/101night/101night_analy.py")
analy = importlib.util.module_from_spec(spec)
print("loading module...", flush=True)
spec.loader.exec_module(analy)
print("module loaded", flush=True)

from pathlib import Path
data_dir = Path("I:/101Night")
mffs = sorted(data_dir.glob("Nathalie-*.mff"))
print(f"Found {len(mffs)} mff dirs", flush=True)
for m in mffs[:3]:
    print(f"  {m.name}", flush=True)
print("=== DONE ===", flush=True)
