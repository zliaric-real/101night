"""
batch_analyze.py — 101-nights 批处理与可视化 (高效版)
========================================================
仅加载 YASA 所需的 3 通道 (E21 + E67 + E219), 大幅加速批处理。
按日期排序，生成睡眠分期和特征波统计可视化。

用法: python batch_analyze.py
"""
import os, sys, re, pickle, warnings, gc, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── 系统内存检查（可选，无 psutil 则跳过） ──
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── YASA 预导入: 首次 import 需要 60-120s (TensorFlow 模型下载/编译)
#     放在顶层让延迟在脚本启动时吸收，避免处理中途"假死" ──
try:
    import yasa  # noqa: F401 — 预加载，后续调用即时完成
except ImportError:
    pass  # 如果 yasa 未安装，process_one_night 中会再次尝试并报错

# ── 路径 ──
DATA_DIR   = Path("I:/101Night")
OUTPUT_DIR = Path("E:/idea/101night")
PLOT_DIR   = OUTPUT_DIR / "sleep_plots"
PLOT_DIR.mkdir(exist_ok=True)
CACHE_PATH = OUTPUT_DIR / "batch_results.pkl"
LOG_PATH   = OUTPUT_DIR / "batch_log.txt"

# ── 常量 ──
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {"Wake":"#fc8d62","N1":"#66c2a5","N2":"#8da0cb","N3":"#a6d854","REM":"#e78ac3"}
STAGE_ORDER = ["Wake","N1","N2","N3","REM"]
WAVE_NAMES  = {0:"Background",1:"Spindle",2:"Slow wave",3:"K-complex",4:"Sawtooth",5:"Vertex sharp",6:"Arousal"}
WAVE_COLORS = {"Background":"#cccccc","Spindle":"#377eb8","Slow wave":"#4daf4a",
               "K-complex":"#984ea3","Sawtooth":"#ff7f00","Vertex sharp":"#a65628","Arousal":"#f781bf"}

EEG_CH = "E21"
EOG_L, EOG_R = "E67", "E219"

# ── 日志 ──
def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

# ══════════════════════════════════════════════════════════
#  扫描
# ══════════════════════════════════════════════════════════
def scan_mff_dirs():
    mff_dirs = sorted(DATA_DIR.glob("Nathalie-*.mff"))
    records = []
    for mff_path in mff_dirs:
        name = mff_path.name
        m = re.match(r"Nathalie-(\d+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.mff", name)
        if m:
            night = int(m.group(1))
            y, mo, d = int(m.group(2)), int(m.group(3)), int(m.group(4))
            hh, mm, ss = int(m.group(5)), int(m.group(6)), int(m.group(7))
            records.append({
                "night": night, "date": f"{y}-{mo:02d}-{d:02d}",
                "datetime": datetime(y,mo,d,hh,mm,ss), "path": str(mff_path), "name": name
            })
    records.sort(key=lambda r: r["datetime"])
    return records


# ── 导入共享 I/O (eeg_io.py) ──
from eeg_io import _read_mff_metadata, _load_channel_data


# ══════════════════════════════════════════════════════════
#  单夜处理 — np.frombuffer + YASA (零内存爆炸)
# ══════════════════════════════════════════════════════════
def process_one_night(rec):
    """YASA 睡眠分期：XML 元数据 + chunked np.frombuffer + 预分配缓冲区

    关键设计：
    - 永远用 XML 解析元数据，不碰 MNE read_raw_egi (mmap 风险)
    - 预分配 2MB 缓冲区 + f.readinto() + np.frombuffer() 替代 np.fromfile()
      参考 numpy/numpy#30777 — 避免 malloc 碎片化导致 OOM
    - CHUNK_VALUES = 500_000，内存峰值恒定 ~2MB
    - 逐夜 gc.collect() 防止 Windows 虚拟内存碎片化
    """
    import mne

    # Always import yasa (Python caches; run_pipeline may have pre-imported it)
    import yasa

    night = rec["night"]
    result = {"night": night, "date": rec["date"], "yasa_ok": False,
              "stages": None, "stage_pct": None, "stage_counts": None,
              "loader": "xml"}

    try:
        # ── Step 1: XML 元数据（永远跳过 MNE） ──
        meta = _read_mff_metadata(rec["path"])
        ch_names = meta["ch_names"]
        sfreq = meta["sfreq"]
        n_channels = meta["n_channels"]
        n_times = meta["n_times"]

        if n_channels == 0 or n_times == 0:
            log(f"  N{night}: empty metadata (ch={n_channels}, n={n_times})")
            return result

        log(f"  N{night}: {n_channels}ch, {sfreq:.0f}Hz, {n_times/sfreq/3600:.1f}h [xml]")

        # ── Step 2: 通道索引 ──
        ch_map = {ch: i for i, ch in enumerate(ch_names)}
        if EEG_CH not in ch_map:
            log(f"  N{night}: {EEG_CH} not found, skip")
            return result

        eeg_idx = ch_map[EEG_CH]
        eogL_idx = ch_map.get(EOG_L)
        eogR_idx = ch_map.get(EOG_R)
        have_eog = eogL_idx is not None and eogR_idx is not None

        # ── Step 3: 预分配缓冲区读取 signal1.bin ──
        # 使用 np.frombuffer + 预分配缓冲区替代 np.fromfile()：
        #   np.fromfile() 每次调用底层 malloc()，循环数千次后 Windows
        #   malloc 碎片化，导致连 1.91MB 的 CHUNK 都分配不出。
        #   参考: numpy/numpy#30777 "add out parameter to fromfile"
        # 预分配一块内存，每次 f.readinto() 写入同一块，完全绕过 malloc。
        signal_path = Path(rec["path"]) / "signal1.bin"
        CHUNK_VALUES = 500_000
        CHUNK_BYTES = CHUNK_VALUES * 4  # float32 = 4 bytes
        eeg_chunks = []
        eogL_chunks = [] if have_eog else None
        eogR_chunks = [] if have_eog else None

        # 预分配一次，全循环复用
        _buf = np.empty(CHUNK_VALUES, dtype=np.float32)
        _buf_mv = memoryview(_buf).cast('B')

        t_read = time.time()
        with open(str(signal_path), "rb") as f:
            while True:
                n_read = f.readinto(_buf_mv)
                if n_read == 0:
                    break
                n_values = n_read // 4
                if n_values < n_channels:  # last partial chunk
                    if n_values == 0:
                        break
                    # use a view of only the valid portion
                    chunk = _buf[:n_values]
                else:
                    chunk = _buf  # reuse the full buffer
                n_samples = n_values // n_channels
                if n_samples == 0:
                    break
                chunk_2d = chunk[:n_samples * n_channels].reshape(n_samples, n_channels)
                eeg_chunks.append(chunk_2d[:, eeg_idx].copy())
                if have_eog:
                    eogL_chunks.append(chunk_2d[:, eogL_idx].copy())
                    eogR_chunks.append(chunk_2d[:, eogR_idx].copy())

        # 拼接 + 转换
        eeg = np.concatenate(eeg_chunks).astype("float64")  # signal1.bin 已是 μV
        del eeg_chunks
        eeg_mb = eeg.nbytes / 1e6

        if have_eog:
            eogL = np.concatenate(eogL_chunks).astype("float64")  # signal1.bin 已是 μV
            eogR = np.concatenate(eogR_chunks).astype("float64")  # signal1.bin 已是 μV
            eog = eogL - eogR
            del eogL, eogR, eogL_chunks, eogR_chunks
            log(f"  N{night}: data loaded — EEG {eeg_mb:.0f}MB, EOG bipolar ({time.time()-t_read:.0f}s)")
            combined = np.vstack([eeg, eog])
            del eog
            info = mne.create_info([EEG_CH, "EOG"], sfreq, ch_types=["eeg", "eog"])
            raw_stage = mne.io.RawArray(combined, info, verbose=False)
            del combined
            eog_arg = "EOG"
        else:
            log(f"  N{night}: data loaded — EEG {eeg_mb:.0f}MB (no EOG) ({time.time()-t_read:.0f}s)")
            raw_stage = mne.io.RawArray(
                eeg[np.newaxis, :],
                mne.create_info([EEG_CH], sfreq, ch_types=["eeg"]),
                verbose=False)
            eog_arg = None

        del eeg

        # ── Step 4: YASA 睡眠分期 ──
        t_yasa = time.time()
        log(f"  N{night}: YASA starting... ({n_times/sfreq/3600:.1f}h, {sfreq:.0f}Hz)")
        sls = yasa.SleepStaging(raw_stage, eeg_name=EEG_CH, eog_name=eog_arg)
        hypno = np.asarray(sls.predict(), dtype=int)  # TF tensor → numpy copy
        del sls, raw_stage
        log(f"  N{night}: YASA predict done ({time.time()-t_yasa:.0f}s)")

        labels_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
        counts = {labels_map.get(s, s): int((hypno == s).sum()) for s in np.unique(hypno)}
        total = len(hypno)
        pct = {k: f"{v/total*100:.1f}%" for k, v in counts.items()}

        result["stages"] = hypno
        result["stage_counts"] = counts
        result["stage_pct"] = pct
        result["yasa_ok"] = True
        result["n_epochs"] = total
        log(f"  N{night}: DONE ({total} epochs) — {pct}")

    except Exception as e:
        log(f"  N{night}: FAILED — {e}")

    # 强制回收，防止 Windows 虚拟内存碎片化
    gc.collect()
    return result


# ══════════════════════════════════════════════════════════
#  可视化
# ══════════════════════════════════════════════════════════

def trim_leading_wake(stages):
    if stages is None or len(stages)==0:
        return stages, 0
    for i, s in enumerate(stages):
        if s != 0:
            return stages[i:], i
    return stages, 0


def plot_hypnogram_all(results):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [r for r in results if r["yasa_ok"] and r["stages"] is not None]
    if not valid:
        log("No YASA data for hypnogram")
        return

    n = len(valid)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*2.2), squeeze=False)
    fig.suptitle("101-nights — Sleep Hypnograms (initial Wake trimmed)", fontsize=14, y=1.01)

    for idx, rec in enumerate(valid):
        ax = axes[idx//ncols][idx%ncols]
        stages, _ = trim_leading_wake(rec["stages"])
        if len(stages) == 0:
            ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes)
            ax.set_title(f"N{rec['night']} {rec['date']}", fontsize=9)
            continue
        colors = [STAGE_COLORS.get(STAGE_NAMES.get(s,"Wake"),"#999") for s in stages]
        ax.bar(range(len(stages)), 1, width=1, color=colors, edgecolor="none")
        ax.set_ylim(0,1); ax.set_yticks([]); ax.set_xlim(0,len(stages))
        nticks = min(6, len(stages)//10+1)
        if nticks>1:
            tp = np.linspace(0,len(stages)-1,nticks).astype(int)
            ax.set_xticks(tp); ax.set_xticklabels([f"{p*30/3600:.1f}h" for p in tp],fontsize=7,rotation=30)
        ax.set_title(f"N{rec['night']} {rec['date']} ({len(stages)}ep)", fontsize=9)

    for i in range(len(valid), nrows*ncols):
        axes[i//ncols][i%ncols].set_visible(False)

    handles = [plt.Rectangle((0,0),1,1,color=STAGE_COLORS[s]) for s in STAGE_ORDER]
    fig.legend(handles, STAGE_ORDER, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout()
    p = PLOT_DIR / "hypnogram_all.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    log(f"Saved: {p}")


def plot_stage_distribution(results):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [r for r in results if r["yasa_ok"] and r["stage_counts"] is not None]
    if not valid: return

    matrix = []
    nights = [r["night"] for r in valid]
    for rec in valid:
        ct = rec["stage_counts"]; t = sum(ct.values())
        matrix.append([ct.get(s,0)/t*100 for s in STAGE_ORDER])
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(14,len(valid)*0.4), 6))
    x = np.arange(len(valid)); bottom = np.zeros(len(valid))
    for i, s in enumerate(STAGE_ORDER):
        ax.bar(x, matrix[:,i], bottom=bottom, color=STAGE_COLORS[s], label=s, width=0.8, edgecolor="white", lw=0.3)
        bottom += matrix[:,i]

    rot = 90 if len(valid)>15 else 45; fs = 7 if len(valid)>20 else 9
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{n}" for n in nights], fontsize=fs, rotation=rot, ha="center")
    ax.set_ylabel("%"); ax.set_title("Sleep Stage Distribution Across Nights", fontsize=14)
    ax.legend(loc="upper right", ncol=5, fontsize=8, frameon=False); ax.set_ylim(0,105)
    plt.tight_layout()
    p = PLOT_DIR / "stage_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    log(f"Saved: {p}")


def plot_summary_dashboard(results):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [r for r in results if r["yasa_ok"] and r["stage_counts"] is not None]
    if not valid: return

    nights = [r["night"] for r in valid]
    dates  = [r["date"] for r in valid]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (0,0) 睡眠效率
    ax = axes[0,0]
    eff = []
    for rec in valid:
        ct = rec["stage_counts"]; t = sum(ct.values())
        eff.append((t - ct.get("Wake",0))/t*100)
    ax.plot(range(len(valid)), eff, "o-", color="#2c7bb6", markersize=6, lw=2, mfc="white")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([f"N{n}" for n in nights], fontsize=7, rotation=90 if len(valid)>15 else 45)
    ax.set_ylabel("Sleep Efficiency (%)"); ax.set_title("Sleep Efficiency", fontsize=12)
    ax.axhline(np.mean(eff), color="red", ls="--", lw=1, label=f"Mean={np.mean(eff):.0f}%")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0,105)

    # (0,1) N3 + REM
    ax = axes[0,1]
    n3p, remp = [], []
    for rec in valid:
        ct = rec["stage_counts"]; t = sum(ct.values())
        n3p.append(ct.get("N3",0)/t*100); remp.append(ct.get("REM",0)/t*100)
    ax.plot(range(len(valid)), n3p, "s-", color="#a6d854", markersize=5, lw=2, label="N3 (Deep)")
    ax.plot(range(len(valid)), remp, "^-", color="#e78ac3", markersize=5, lw=2, label="REM")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([f"N{n}" for n in nights], fontsize=7, rotation=90 if len(valid)>15 else 45)
    ax.set_ylabel("%"); ax.set_title("N3 & REM Trends", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (1,0) 每夜时长
    ax = axes[1,0]
    hours = [r.get("n_epochs",0)*30/3600 for r in valid]
    ax.bar(range(len(valid)), hours, color="#8da0cb", edgecolor="white")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([f"N{n}" for n in nights], fontsize=7, rotation=90 if len(valid)>15 else 45)
    ax.set_ylabel("Hours"); ax.set_title("Recording Duration per Night", fontsize=12)
    ax.axhline(np.mean(hours), color="red", ls="--", lw=1, label=f"Mean={np.mean(hours):.1f}h")
    ax.legend(fontsize=8)

    # (1,1) 统计表
    ax = axes[1,1]; ax.axis("off")
    txt = "=== 101-nights Summary ===\n\n"
    txt += f"Nights scanned : {len(results)}\n"
    txt += f"YASA success  : {len(valid)}\n\n"
    wake_pct = [(r["stage_counts"].get("Wake",0)/sum(r["stage_counts"].values())*100) for r in valid]
    n3_pct   = [(r["stage_counts"].get("N3",0)/sum(r["stage_counts"].values())*100) for r in valid]
    rem_pct  = [(r["stage_counts"].get("REM",0)/sum(r["stage_counts"].values())*100) for r in valid]
    txt += f"Mean Wake : {np.mean(wake_pct):5.1f}% ± {np.std(wake_pct):.1f}%\n"
    txt += f"Mean N3   : {np.mean(n3_pct):5.1f}% ± {np.std(n3_pct):.1f}%\n"
    txt += f"Mean REM  : {np.mean(rem_pct):5.1f}% ± {np.std(rem_pct):.1f}%\n\n"
    txt += f"Mean duration: {np.mean(hours):.1f}h ± {np.std(hours):.1f}h\n\n"
    txt += "Nights:\n"
    for r in results:
        s = "✓" if r["yasa_ok"] else "✗"
        txt += f"  N{r['night']:3d}  {r['date']}  [{s}]\n"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, fontfamily="monospace", va="top")

    plt.suptitle("101-nights — Sleep Analysis Dashboard", fontsize=14, y=1.02)
    plt.tight_layout()
    p = PLOT_DIR / "summary_dashboard.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    log(f"Saved: {p}")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════
def main():
    # 清空日志
    with open(LOG_PATH, "w") as f: f.write("")

    log("="*60)
    log("101-nights Batch Analyzer (Efficient)")
    log(f"Data: {DATA_DIR}  |  Output: {OUTPUT_DIR}")
    log("="*60)

    # ── 内存状况检查 ──
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        log(f"System memory: {mem.available/1e9:.1f} GB available "
            f"/ {mem.total/1e9:.1f} GB total ({mem.percent}% used)")
        if mem.available < 4 * 1024**3:  # < 4 GB
            log("⚠ WARNING: Less than 4 GB available — batch may fail!")
            log("  Close other Python processes (esp. 101night_analy.py) "
                "and retry.")
    else:
        log("(install psutil for memory monitoring: pip install psutil)")

    records = scan_mff_dirs()
    log(f"Found {len(records)} nights (date-sorted)")
    for i, r in enumerate(records):
        log(f"  {i+1:2d}. N{r['night']:3d}  {r['date']}")

    # 加载缓存
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            results = pickle.load(f)
        done_nights = {r["night"] for r in results if r["yasa_ok"]}
        log(f"Loaded {len(results)} cached results ({len(done_nights)} done)")
    else:
        results = []
        done_nights = set()

    # 处理
    for i, rec in enumerate(records):
        if rec["night"] in done_nights:
            log(f"[{i+1}/{len(records)}] N{rec['night']} — SKIP (cached)")
            continue

        log(f"\n[{i+1}/{len(records)}] N{rec['night']} {rec['date']}")
        result = process_one_night(rec)
        results.append(result)
        log(f"  [CACHE] Saving {len(results)} results...")
        # 逐夜保存，崩溃不丢进度
        _seen = {r["night"]: r for r in results}
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(list(_seen.values()), f)
        log(f"  [CACHE] Saved.")
        # 内存心跳 — 监控是否有泄漏
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            log(f"  [MEM] {mem.available/1e9:.1f} GB free ({mem.percent}% used)")

    # 去重 (缓存+新结果可能有重复)
    seen = {}
    for r in results:
        seen[r["night"]] = r
    results = list(seen.values())
    results.sort(key=lambda r: records_dict.get(r["night"], datetime.max))

    # 保存最终结果
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(results, f)

    ok = sum(1 for r in results if r["yasa_ok"])
    log(f"\nDone: {ok}/{len(results)} YASA successful")

    # 可视化
    log("\n" + "="*60)
    log("Generating plots...")
    log("="*60)

    plot_hypnogram_all(results)
    plot_stage_distribution(results)
    plot_summary_dashboard(results)

    log(f"\nAll done! Results: {CACHE_PATH}")
    log(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    records_dict = {}
    for r in scan_mff_dirs():
        records_dict[r["night"]] = r["datetime"]
    main()
