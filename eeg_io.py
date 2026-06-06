# -*- coding: utf-8 -*-
"""
eeg_io.py — EGI .mff 共享 I/O (元数据解析 + chunked 通道加载)
============================================================
被 feature_101night_analy.py、batch_analyze.py、各 step 脚本 import。

函数:
  _read_mff_metadata(mff_path_str) → {ch_names, n_channels, sfreq, n_times}
  _load_channel_data(mff_path_str, n_channels, channel_indices) → {idx: array}
  _load_channel_data_slice(mff_path_str, n_channels, indices, start, n) → {idx: array}
"""

import numpy as np
from pathlib import Path
import gc


# ═══════════════════════════════════════════════════════════
#  MFF metadata parser (bypass MNE EGI reader memory issues)
# ═══════════════════════════════════════════════════════════
def _read_mff_metadata(mff_path_str):
    """Parse .mff metadata from sensorLayout.xml + signal1.bin.
    Returns: {ch_names, n_channels, sfreq, n_times}
    """
    import xml.etree.ElementTree as ET

    mff = Path(mff_path_str)

    # ── sensorLayout.xml → channel names ──
    layout_path = mff / "sensorLayout.xml"
    if not layout_path.exists():
        raise FileNotFoundError(f"sensorLayout.xml not found in {mff}")

    ltree = ET.parse(str(layout_path))
    lroot = ltree.getroot()
    lns = lroot.tag.split('}')[0] + '}' if '}' in lroot.tag else ''

    sensors_el = lroot.find(f'{lns}sensors')
    if sensors_el is None:
        raise ValueError("No <sensors> in sensorLayout.xml")

    channels = []
    for sensor in sensors_el.findall(f'{lns}sensor'):
        snum_el = sensor.find(f'{lns}number')
        stype_el = sensor.find(f'{lns}type')
        sname_el = sensor.find(f'{lns}name')
        sname = sname_el.text.strip() if sname_el is not None and sname_el.text else ""
        stype = int(stype_el.text) if stype_el is not None else 0
        snum = int(snum_el.text) if snum_el is not None else 0
        if sname:
            channels.append(sname)
        elif stype == 0:
            channels.append(f"E{snum}")

    # ── n_channels from signal1.bin size ──
    signal_path = mff / "signal1.bin"
    file_size = signal_path.stat().st_size
    total_values = file_size // 4
    n_channels = 260
    if total_values % 260 != 0:
        for n in [257, 256, 261, 259, 258, 269]:
            if total_values % n == 0:
                n_channels = n
                break
    n_times = total_values // n_channels

    # truncate/pad channel names to match
    if len(channels) > n_channels:
        channels = channels[:n_channels]
    while len(channels) < n_channels:
        channels.append(f"AUX{len(channels)+1}")

    # ── sfreq: EGI 256 hardware is 250 Hz ──
    sfreq = 250.0
    for info_name in ["info1.xml", "info.xml"]:
        ip = mff / info_name
        if not ip.exists():
            continue
        itree = ET.parse(str(ip))
        iroot = itree.getroot()
        ins = iroot.tag.split('}')[0] + '}' if '}' in iroot.tag else ''
        sr = iroot.find(f'{ins}sampRate')
        if sr is None:
            sr = iroot.find(f'.//{ins}sampRate')
        if sr is not None and sr.text:
            sfreq_xml = float(sr.text)
            if sfreq_xml > 0 and 100 <= sfreq_xml <= 2000:
                sfreq = sfreq_xml
            break

    return {'ch_names': channels, 'n_channels': n_channels,
            'sfreq': sfreq, 'n_times': n_times}


def _load_channel_data(mff_path_str, n_channels, channel_indices,
                        n_times=None, sfreq=None, filter_params=None):
    """Load selected channels from signal1.bin using chunked frombuffer.

    Uses pre-allocated buffer + f.readinto() + np.frombuffer() to avoid
    malloc fragmentation from repeated np.fromfile() calls.
    Reference: numpy/numpy#30777

    Returns:
        dict mapping channel_idx -> 1D float64 array (μV, NA400 stores μV natively)
    """
    signal_path = Path(mff_path_str) / "signal1.bin"

    CHUNK_VALUES = 500_000  # ~2 MB
    _buf = np.empty(CHUNK_VALUES, dtype=np.float32)
    _buf_mv = memoryview(_buf).cast('B')

    channel_chunks = {idx: [] for idx in channel_indices}

    with open(str(signal_path), "rb") as f:
        while True:
            n_read = f.readinto(_buf_mv)
            if n_read == 0:
                break
            n_values = n_read // 4
            if n_values < n_channels:
                if n_values == 0:
                    break
                chunk = _buf[:n_values]
            else:
                chunk = _buf
            n_samples = n_values // n_channels
            if n_samples == 0:
                break
            chunk_2d = chunk[:n_samples * n_channels].reshape(n_samples, n_channels)
            for idx in channel_indices:
                channel_chunks[idx].append(chunk_2d[:, idx].copy())

    result = {}
    for idx in channel_indices:
        data = np.concatenate(channel_chunks[idx]).astype("float64")  # signal1.bin 已是 μV
        result[idx] = data

    gc.collect()
    return result


def _load_channel_data_slice(mff_path_str, n_channels, channel_indices,
                              start_sample, n_samples, sfreq=None,
                              filter_params=None):
    """Load a time slice of selected channels from signal1.bin.

    Reads only the specified sample range from the interleaved binary file.
    Used for source localization to avoid loading the entire recording.

    signal1.bin layout: [s0_c0, s0_c1, ..., s0_c259, s1_c0, ...]
    Each value is float32 (4 bytes).

    Returns:
        dict mapping channel_index -> 1D float64 array (μV)
    """
    signal_path = Path(mff_path_str) / "signal1.bin"
    file_size = signal_path.stat().st_size

    bytes_per_sample = n_channels * 4  # float32
    start_byte = start_sample * bytes_per_sample
    bytes_to_read = n_samples * bytes_per_sample

    # Clamp to file bounds
    if start_byte >= file_size:
        return {}
    if start_byte + bytes_to_read > file_size:
        bytes_to_read = file_size - start_byte

    CHUNK_VALUES = 500_000  # ~2 MB
    _buf = np.empty(CHUNK_VALUES, dtype=np.float32)
    _buf_mv = memoryview(_buf).cast('B')

    channel_chunks = {idx: [] for idx in channel_indices}
    bytes_read = 0

    with open(str(signal_path), "rb") as f:
        f.seek(start_byte)
        while bytes_read < bytes_to_read:
            remaining = bytes_to_read - bytes_read
            to_read = min(CHUNK_VALUES * 4, remaining)
            if to_read == 0:
                break
            n_read = f.readinto(_buf_mv[:to_read])
            if n_read == 0:
                break
            bytes_read += n_read
            n_values = n_read // 4
            if n_values < n_channels:
                break
            n_samples_chunk = n_values // n_channels
            chunk_2d = _buf[:n_samples_chunk * n_channels].reshape(
                n_samples_chunk, n_channels)
            for idx in channel_indices:
                channel_chunks[idx].append(chunk_2d[:, idx].copy())

    result = {}
    for idx in channel_indices:
        chunks = channel_chunks[idx]
        if chunks:
            data = np.concatenate(chunks).astype("float64")  # signal1.bin 已是 μV
            result[idx] = data

    del channel_chunks; gc.collect()
    return result


def detect_n_channels(mff_path_str):
    """从 signal1.bin 文件大小自动检测实际通道数。

    与 _read_mff_metadata 不同: MNE 报告的名称列表可能包含
    不在文件中的虚拟通道 (VREF, DIN1 等), 因此必须直接计算
    signal1.bin 的整除通道数。

    Returns:
        int: 实际存储在 signal1.bin 中的通道数
    """
    signal_path = Path(mff_path_str) / "signal1.bin"
    file_size = signal_path.stat().st_size
    total_values = file_size // 4
    # 优先尝试 257 (常见), 然后 260, 256, ...
    for n in [257, 260, 256, 261, 259, 258, 269]:
        if total_values % n == 0:
            return n
    # 未找到整除 → 默认 257
    return 257
