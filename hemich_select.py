# -*- coding: utf-8 -*-
"""
hemich_select.py — EGI 256 半球代表通道选择 (独立脚本, 纯 numpy)
================================================================
解析 sensorLayout.xml，通过空间分位法选出每半球 ≤5 个代表通道，
覆盖额叶/中央/顶叶/颞叶/枕叶。不依赖 sklearn。

EGI 256 电极位置是固定硬件参数，一次计算全夜通用。
输出 hemi_channels.pkl 供 feature_101night_analy.py 加载。

用法:
  python hemich_select.py --mff <sample.mff>
"""

import sys
import pickle
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PKL = PROJECT_DIR / "hemi_channels.pkl"


def parse_sensor_coords(mff_path):
    """Parse sensorLayout.xml → coords (n_eeg, 3) + ch_names list."""
    mff = Path(mff_path)
    layout = mff / "sensorLayout.xml"
    tree = ET.parse(str(layout))
    root = tree.getroot()
    ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''

    sensors = root.find(f'{ns}sensors').findall(f'{ns}sensor')
    coords = []
    ch_names = []
    for sensor in sensors:
        stype = int(sensor.find(f'{ns}type').text) if sensor.find(f'{ns}type') is not None else 0
        if stype != 0:
            continue
        snum = int(sensor.find(f'{ns}number').text)
        ch_names.append(f"E{snum}")
        coords.append([
            float(sensor.find(f'{ns}x').text),
            float(sensor.find(f'{ns}y').text),
            float(sensor.find(f'{ns}z').text),
        ])
    # Normalize to unit sphere
    coords_arr = np.array(coords)
    norms = np.linalg.norm(coords_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return coords_arr / norms, ch_names


def label_region(x, y, z):
    """Label a channel by approximate anatomical region (unit-sphere coords)."""
    if y > 0.3:
        return 'F'   # frontal / anterior
    elif y < -0.5:
        return 'O'   # occipital / posterior
    elif abs(x) > 0.5:
        return 'T'   # temporal / lateral
    elif y <= -0.1:
        return 'P'   # parietal
    else:
        return 'C'   # central


REGION_NAMES = {'F': '额叶', 'C': '中央', 'P': '顶叶',
                'O': '枕叶', 'T': '颞叶'}


def select_hemi_channels(coords, ch_names, max_per_hemi=5):
    """Select representative channels per hemisphere by spatial coverage.

    Algorithm: sort channels by anterior→posterior (y), divide into
    k bins, pick channel closest to bin median on (y,z) plane.
    Ensures coverage across frontal→occipital axis. No sklearn.
    """
    # Sort by y (anterior → posterior)
    order = np.argsort(coords[:, 1])[::-1]
    n = len(coords)
    k = min(max_per_hemi, n)
    if k <= 1:
        return [ch_names[order[0]]]

    # Divide sorted channels into k equal bins, pick median of each
    picks = []
    for i in range(k):
        start = i * n // k
        end = (i + 1) * n // k
        bin_indices = order[start:end]
        if len(bin_indices) == 0:
            continue
        # Pick channel closest to bin median in (y, z) space
        bin_coords = coords[bin_indices]
        median_pt = np.median(bin_coords[:, 1:], axis=0)  # (y, z)
        dists = np.linalg.norm(bin_coords[:, 1:] - median_pt, axis=1)
        best = bin_indices[np.argmin(dists)]
        picks.append(best)

    return [ch_names[p] for p in picks]


def select_hemispheric_channels(mff_path, max_per_hemi=5):
    """Main entry: select channels per hemisphere, return result dict."""
    coords, ch_names = parse_sensor_coords(mff_path)

    left_mask = coords[:, 0] < 0
    right_mask = coords[:, 0] >= 0

    result = {'max_per_hemi': max_per_hemi,
              'source_mff': str(Path(mff_path).name)}

    for mask, side in [(left_mask, 'L'), (right_mask, 'R')]:
        hc = coords[mask]
        hn = [ch_names[i] for i in range(len(ch_names)) if mask[i]]
        if len(hn) == 0:
            result[f'{side.lower()}_names'] = []
            continue
        picked = select_hemi_channels(hc, hn, max_per_hemi)
        result[f'{side.lower()}_names'] = picked

    # Combine + sort
    all_names = result['l_names'] + result['r_names']
    result['hemi_names'] = all_names
    result['n_hemi'] = len(all_names)

    # Region labels
    name_to_xyz = {ch_names[i]: coords[i] for i in range(len(coords))}
    result['regions'] = {ch: label_region(*name_to_xyz[ch])
                         for ch in all_names if ch in name_to_xyz}

    return result


def print_summary(result):
    """Print human-readable summary."""
    print(f"\n{'='*60}")
    print(f"半球代表通道选择 (max_per_hemi={result['max_per_hemi']})")
    print(f"来源: {result['source_mff']}")
    print(f"{'='*60}")
    print(f"总计: {result['n_hemi']} 通道")

    for side in ('L', 'R'):
        names = result[f'{side.lower()}_names']
        print(f"\n  {side}半球 ({len(names)} 通道):")
        for ch in names:
            reg = result['regions'].get(ch, '?')
            print(f"    {ch:>5}  → {reg} ({REGION_NAMES.get(reg, '?')})")

    from collections import Counter
    rc = Counter(result['regions'].values())
    print(f"\n  区域分布: {dict(rc)}")


def main():
    parser = argparse.ArgumentParser(
        description='EGI 256 半球代表通道选择 (纯 numpy, 无 sklearn)')
    parser.add_argument('--mff',
                        default=str(PROJECT_DIR /
                                    'Nathalie-40_20171011_121248.mff'),
                        help='样本 .mff 路径')
    parser.add_argument('--max-per-hemi', type=int, default=5,
                        help='每半球最多通道数 (默认 5)')
    parser.add_argument('--output', default=str(DEFAULT_PKL),
                        help='输出 .pkl 路径')
    args = parser.parse_args()

    mff_path = Path(args.mff)
    if not mff_path.exists():
        print(f"✗ .mff 不存在: {mff_path}")
        return 1

    result = select_hemispheric_channels(
        str(mff_path), max_per_hemi=args.max_per_hemi)
    print_summary(result)

    out_path = Path(args.output)
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\n✓ 已保存: {out_path}")
    return 0


def load_hemich(pkl_path=None):
    """Load pre-computed hemispheric channels from pkl."""
    path = Path(pkl_path) if pkl_path else DEFAULT_PKL
    if not path.exists():
        raise FileNotFoundError(
            f"hemi_channels.pkl 不存在: {path}\n"
            f"请先运行: python hemich_select.py --mff <sample.mff>")
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    sys.exit(main())
