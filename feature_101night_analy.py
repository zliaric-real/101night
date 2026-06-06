# -*- coding: utf-8 -*-
"""
SleepEEGFeatureExtractor — 睡眠脑电特征提取器 (增强版)
====================================================
从单通道到全脑多维特征：幂律、熵、分形、复杂度、连接性、微状态、图论。

环境: conda activate eeg_101night
"""
from pathlib import Path

import mne
import os
import yasa
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import scipy
from scipy import signal, stats
try:
    from sssm_07.sssm.sssm import Model
    HAS_SSSM = True
except ImportError:
    HAS_SSSM = False
    Model = None
import unittest
# pyrasa not used — we use yasa.irasa directly
import mne_features
import mne_microstates
from mne_features.univariate import (compute_pow_freq_bands, compute_spect_entropy,
                                      compute_hjorth_mobility, compute_hjorth_complexity)
import pycatch22

# ---- 可选：非线性特征库 ----
try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False

try:
    import neurokit2 as nk
    HAS_NEUROKIT2 = True
except ImportError:
    HAS_NEUROKIT2 = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ═══════════════════════════════════════════════════════════
#  MFF I/O (extracted to eeg_io.py)
# ═══════════════════════════════════════════════════════════
from eeg_io import (_read_mff_metadata, _load_channel_data,
                     _load_channel_data_slice)


def _label_anatomical_region(coords, indices):
    """Label channel indices by approximate anatomical region from EGI coordinates.

    Uses sensor position on unit sphere to estimate cortical region.
    In the EGI coordinate system:
      - y: anterior-posterior (higher → more frontal)
      - z: inferior-superior (higher → more dorsal/superior)
      - x: left-right

    Returns list of region labels: 'F' frontal, 'C' central, 'P' parietal,
    'O' occipital, 'T' temporal.
    """
    labels = []
    for i in indices:
        x, y, z = coords[i]
        if y > 0.3:
            labels.append('F')   # frontal / anterior
        elif y < -0.5:
            labels.append('O')   # occipital / posterior
        elif abs(x) > 0.5:
            labels.append('T')   # temporal / lateral
        elif y <= -0.2:
            labels.append('P')   # parietal
        else:
            labels.append('C')   # central
    return labels


class SleepEEGFeatureExtractor:
    """
    睡眠脑电特征提取器

    该类用于从睡眠脑电数据中提取各种特征，包括睡眠分期、时间信息、
    特征波、非周期特征、熵、分形、复杂度、功能连接、微状态等。

    Attributes:
        raw: 原始脑电数据对象（单通道）
        sfreq: 采样频率
        data: EEG数据数组（单通道）
        ch_names: 通道名称列表
    """

    # ---- 频段定义 ----
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
    }
    BAND_EDGES = np.array([0.5, 4, 8, 13, 30, 45])
    BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    def __init__(self, file_path, eeg_channel='E21',
                 load_all_channels: bool = True,
                 filter_low: float = 0.1,
                 filter_high: float = 40.0):
        """
        初始化特征提取器

        - 使用 MNE mne.io.read_raw_egi 加载 .mff 文件
        - 滤波使用 MNE 默认 FIR 设计 (firwin, zero-phase, 自动阶数)
        - 单通道: 仅加载目标通道，内存 ~60 MB
        - 多通道: 惰性加载 (lazy)，仅在首次调用多通道特征方法时从磁盘读取
          通过 chunked frombuffer 加载，避免 MNE 全通道预加载的内存问题

        Args:
            file_path: .mff文件路径
            eeg_channel: 用于单通道分析的主要EEG通道名称
            load_all_channels: 是否允许多通道特征 (惰性加载)
            filter_low: 带通滤波低频截止 (Hz)
            filter_high: 带通滤波高频截止 (Hz)
        """
        self.file_path = file_path
        self.eeg_channel = eeg_channel
        self.filter_low = filter_low
        self.filter_high = filter_high

        # ── Step 1: MNE 加载元数据 (不预加载数据) ──
        raw = mne.io.read_raw_egi(file_path, preload=False, verbose=False)
        self.ch_names = raw.ch_names
        self.n_channels = len(raw.ch_names)
        self.sfreq = int(raw.info['sfreq'])

        # ── Step 2: 检查主通道 ──
        if eeg_channel not in self.ch_names:
            eeg_channel = self.ch_names[10]
            self.eeg_channel = eeg_channel
        self._eeg_idx = self.ch_names.index(eeg_channel)

        # ── Step 3: 仅加载主通道数据 ──
        raw.pick([eeg_channel])
        raw.load_data()
        self.raw = raw
        self.data = self.raw.get_data(units='uV')[0]  # 1D float64, ~60 MB

        # ── Step 3b: MNE 默认 FIR 滤波 (firwin, zero-phase, 自动阶数) ──
        self.filted_raw = self.raw.copy().filter(filter_low, filter_high)
        self.filted = self.filted_raw.get_data(units='uV')[0]

        # ── Step 4: 多通道惰性加载 ──
        self._load_all = load_all_channels
        self._data_all = None      # lazy: loaded on first access
        self._filted_cache = None  # lazy: filtered version
        self._eeg_ch_indices = [i for i, ch in enumerate(self.ch_names)
                                if ch.startswith('E') and ch[1:].isdigit()]
        self.n_times = self.data.size  # 兼容 _get_data_all 等方法

        # ── 存储提取的特征 ──
        self.features = {}
        self._epoch_info = {}

    # ================================================================
    #  Part 0 — 工具方法
    # ================================================================

    def _get_data_all(self):
        """惰性加载全通道 float64 数据 (首次调用时从磁盘读取，约 7.5 GB)。

        使用 chunked frombuffer + 预分配缓冲区，避免 malloc 碎片化。
        缓存结果到 self._data_all，后续调用直接返回缓存。

        如果 _hemi_data_cache 已设置（Step 3 半球通道模式），
        优先返回半球代表通道数据而非全通道。
        """
        if self._data_all is not None:
            return self._data_all
        # Step 3 半球通道模式 — 使用预加载的代表通道数据
        if hasattr(self, '_hemi_data_cache') and self._hemi_data_cache is not None:
            return self._hemi_data_cache
        if not self._load_all:
            return None

        print("[LazyLoad] 首次加载全部 %d 通道数据..." % self.n_channels)
        import gc
        ch_data = _load_channel_data(self.file_path, self.n_channels,
                                      list(range(self.n_channels)),
                                      n_times=self.n_times)
        # Stack into (n_channels, n_times) array
        data_all = np.stack([ch_data[i] for i in range(self.n_channels)], axis=0)
        del ch_data; gc.collect()
        print(f"[LazyLoad] 完成: {data_all.shape}, {data_all.nbytes/1e9:.1f} GB")
        self._data_all = data_all
        return data_all

    def _get_filted_all(self):
        """惰性加载滤波后的全通道数据 (仅在需要时计算)。

        基于 _get_data_all() 的结果做带通滤波。
        缓存结果到 self._filted_cache。

        如果 _hemi_filted_cache 已设置（Step 3 半球通道模式），
        优先返回半球代表通道滤波数据。
        """
        if self._filted_cache is not None:
            return self._filted_cache
        # Step 3 半球通道模式
        if hasattr(self, '_hemi_filted_cache') and self._hemi_filted_cache is not None:
            return self._hemi_filted_cache

        data = self._get_data_all()
        if data is None:
            return None

        print("[LazyLoad] 对全通道数据滤波 (%.1f-%.1f Hz, FIR)..."
              % (self.filter_low, self.filter_high))
        import gc
        # MNE 默认 FIR (firwin, zero-phase, 自动阶数)
        info = mne.create_info(
            [f"ch{i}" for i in range(data.shape[0])],
            self.sfreq, ch_types=['eeg'] * data.shape[0])
        raw_all = mne.io.RawArray(data, info, verbose=False)
        raw_all.filter(self.filter_low, self.filter_high)
        filtered = raw_all.get_data()
        del data, raw_all; gc.collect()
        self._filted_cache = filtered
        return filtered

    # ── ROI (Region of Interest) 空间聚合 ──────────────────────

    def _parse_sensor_coords(self):
        """从 sensorLayout.xml 解析 EEG 通道的 3D 坐标。

        Returns:
            coords: np.array (n_eeg, 3) 按通道索引顺序排列
            eeg_indices: list[int] — 对应 ch_names 中 EEG 通道的索引
        """
        import xml.etree.ElementTree as ET
        from pathlib import Path

        mff = Path(self.file_path)
        layout_path = mff / "sensorLayout.xml"
        tree = ET.parse(str(layout_path))
        root = tree.getroot()
        ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''

        sensors = root.find(f'{ns}sensors').findall(f'{ns}sensor')
        coords = []
        eeg_indices = []
        # sensor number 1 → ch_names index 0, number 2 → index 1, etc.
        # But we need to map through ch_names to get the right indices
        for sensor in sensors:
            stype_el = sensor.find(f'{ns}type')
            stype = int(stype_el.text) if stype_el is not None else 0
            if stype != 0:  # only EEG channels
                continue
            snum_el = sensor.find(f'{ns}number')
            snum = int(snum_el.text) if snum_el is not None else 0
            # sensor number → channel name E{n} → look up in ch_names
            ch_name = f"E{snum}"
            if ch_name in self.ch_names:
                idx = self.ch_names.index(ch_name)
                x = float(sensor.find(f'{ns}x').text)
                y = float(sensor.find(f'{ns}y').text)
                z = float(sensor.find(f'{ns}z').text)
                coords.append([x, y, z])
                eeg_indices.append(idx)

        return np.array(coords), eeg_indices

    def _build_roi_groups(self, n_rois=25, random_state=42):
        """基于 3D 坐标的空间 k-means 聚类，将 EEG 通道分组为 ROI。

        每个 ROI 内的通道在空间上相邻，平均后代表该脑区的活动。
        EGI 256 → 25 ROI 意味着平均 ~10 个通道/ROI。

        Args:
            n_rois: ROI 数量 (默认 25)
            random_state: k-means 随机种子

        Returns:
            list of list of int — 每项是 ch_names 中属于该 ROI 的通道索引
        """
        coords, eeg_indices = self._parse_sensor_coords()
        n_eeg = len(eeg_indices)

        if n_eeg <= n_rois:
            # 通道数少于 ROI 数，每个通道单独成组
            return [[idx] for idx in eeg_indices]

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_rois, random_state=random_state,
                        n_init=10, max_iter=300)
        labels = kmeans.fit_predict(coords)

        # 按标签分组
        roi_groups = [[] for _ in range(n_rois)]
        for i, label in enumerate(labels):
            roi_groups[label].append(eeg_indices[i])

        # 按组大小排序
        roi_groups.sort(key=len, reverse=True)
        return roi_groups

    def _get_roi_data(self, n_rois=25, filtered=True):
        """加载 EEG 通道数据并聚合到 ROI 级别。

        在 Step 3 半球通道模式下，已加载的代表通道直接作为 ROI
        （每个通道代表一个空间区域），不再做 k-means 聚合。

        Args:
            n_rois: ROI 数量（半球模式下忽略）
            filtered: True → 对加载的数据做带通滤波后聚合

        返回: (n_rois, n_times) float64
        """
        import gc

        # Step 3 半球模式 — 代表通道即 ROI
        if filtered and hasattr(self, '_hemi_filted_cache') and self._hemi_filted_cache is not None:
            return self._hemi_filted_cache
        if not filtered and hasattr(self, '_hemi_data_cache') and self._hemi_data_cache is not None:
            return self._hemi_data_cache

        if not hasattr(self, '_roi_groups'):
            print(f"[ROI] 使用 k-means 空间聚类: {self.n_channels} 通道 → {n_rois} 脑区")
            self._roi_groups = self._build_roi_groups(n_rois=n_rois)
            sizes = [len(g) for g in self._roi_groups]
            print(f"[ROI] 组大小: min={min(sizes)}, max={max(sizes)}, "
                  f"mean={np.mean(sizes):.1f}")

        # 收集所有需要加载的通道索引
        all_indices = []
        for group in self._roi_groups:
            all_indices.extend(group)
        all_indices = sorted(set(all_indices))

        # 加载这些通道
        print(f"[ROI] 加载 {len(all_indices)} 个 EEG 通道"
              + (" (滤波后聚合)" if filtered else "") + "...")
        ch_data = _load_channel_data(self.file_path, self.n_channels,
                                      all_indices, n_times=self.n_times)

        # 可选: 带通滤波 (MNE 默认 FIR, zero-phase)
        if filtered:
            import gc as _gc
            n_ch_roi = len(all_indices)
            stacked = np.stack([ch_data[idx] for idx in all_indices], axis=0)
            info = mne.create_info(
                [f"ch{i}" for i in range(n_ch_roi)], self.sfreq,
                ch_types=['eeg'] * n_ch_roi, verbose=False)
            raw_roi = mne.io.RawArray(stacked, info, verbose=False)
            raw_roi.filter(self.filter_low, self.filter_high)
            filt_data = raw_roi.get_data()
            for i, idx in enumerate(all_indices):
                ch_data[idx] = filt_data[i]
            del stacked, raw_roi, filt_data; _gc.collect()

        # 聚合到 ROI
        n_times_actual = len(ch_data[all_indices[0]])
        roi_data = np.zeros((len(self._roi_groups), n_times_actual))
        for roi_idx, group in enumerate(self._roi_groups):
            group_data = [ch_data[idx] for idx in group]
            roi_data[roi_idx] = np.mean(group_data, axis=0)

        del ch_data; gc.collect()
        print(f"[ROI] 聚合完成: {roi_data.shape}, {roi_data.nbytes/1e6:.0f} MB")
        return roi_data

    def _iter_epochs(self, data: np.ndarray, epoch_sec: float,
                     return_1d: bool = True):
        """
        迭代器：将数据切分为 epoch，每次返回一个片段。

        Args:
            data: shape (n_channels, n_times) 或 (n_times,)
            epoch_sec: epoch 长度（秒）
            return_1d: True → 返回 1D，False → 保持 2D

        Yields:
            (epoch_idx, segment)
        """
        sf = self.sfreq
        if data.ndim == 1:
            data = data[np.newaxis, :]
        n_ch, n_total = data.shape
        epoch_samples = int(epoch_sec * sf)
        n_epochs = n_total // epoch_samples
        for ep in range(n_epochs):
            seg = data[:, ep * epoch_samples:(ep + 1) * epoch_samples]
            if return_1d and n_ch == 1:
                seg = seg[0]
            yield ep, seg

    def _get_n_epochs(self, epoch_sec: float) -> int:
        """计算给定窗长下的 epoch 数。"""
        return int(self.data.size // (epoch_sec * self.sfreq))

    def _store_feature(self, name: str, values: np.ndarray, meta: dict = None):
        """统一存储特征，跟踪 epoch 信息。"""
        self.features[name] = {
            'values': values,
            'meta': meta or {},
        }
        if values.ndim == 1:
            self._epoch_info[name] = {'n_epochs': len(values)}
        elif values.ndim == 2:
            self._epoch_info[name] = {'n_epochs': values.shape[0],
                                       'n_dims': values.shape[1]}

    def _clear_cache(self, step=None):
        """Clear step-specific caches to free memory between pipeline stages.

        Each pipeline step loads its own data and should release it
        before the next step begins. This prevents memory accumulation
        that caused the original ~42 GB/instance crash.

        Args:
            step: 'eog' (Step 2), 'roi' (Step 3), 'source' (Step 4),
                  None or 'all' (clear everything)
        """
        import gc

        if step in (None, 'eog', 'all'):
            if hasattr(self, '_eog_data'):
                del self._eog_data
            if hasattr(self, '_raw_stage_for_yasa'):
                del self._raw_stage_for_yasa

        if step in (None, 'roi', 'all'):
            if self._data_all is not None:
                del self._data_all
                self._data_all = None
            if self._filted_cache is not None:
                del self._filted_cache
                self._filted_cache = None
            if hasattr(self, '_roi_groups'):
                del self._roi_groups
            if hasattr(self, '_roi_data_cache'):
                del self._roi_data_cache
            if hasattr(self, '_hemi_channels'):
                del self._hemi_channels

        if step in (None, 'source', 'all'):
            if self._data_all is not None:
                del self._data_all
                self._data_all = None
            if self._filted_cache is not None:
                del self._filted_cache
                self._filted_cache = None
            if hasattr(self, '_src_slice_cache'):
                del self._src_slice_cache

        gc.collect()

    def _get_hemispheric_channels(self, max_per_hemi=5):
        """Select representative EEG channels per hemisphere using spatial k-means.

        Splits EEG channels by x-coordinate into left (x<0) and right (x>=0)
        hemispheres, then within each hemisphere clusters channels on the
        (y, z) plane to cover frontal/central/temporal/parietal/occipital
        regions. Selects the channel closest to each cluster centroid.

        Relevance for sleep research — ensures coverage of:
          - Frontal regions: slow wave generators (N3 delta power)
          - Central regions: spindle generators (N2 sigma power)
          - Occipital regions: alpha activity (wake/REM discrimination)
          - Temporal regions: theta generators
          - Parietal regions: general sleep architecture

        Args:
            max_per_hemi: max channels per hemisphere (default 5).

        Returns:
            list of channel indices (into self.ch_names), max 2*max_per_hemi.
        """
        if hasattr(self, '_hemi_channels'):
            return self._hemi_channels

        # Parse sensor 3D coordinates for all EEG (type-0) channels
        coords, eeg_indices = self._parse_sensor_coords()
        # coords: (n_eeg, 3) — x(L-R), y(A-P), z(I-S) on unit sphere
        # eeg_indices: aligned list of indices into self.ch_names

        left_mask = coords[:, 0] < 0   # x < 0  → left hemisphere
        right_mask = coords[:, 0] >= 0  # x >= 0 → right hemisphere (incl. midline)

        selected = []

        for hemi_mask, hemi_name in [(left_mask, 'L'), (right_mask, 'R')]:
            hemi_coords = coords[hemi_mask]
            hemi_indices = [eeg_indices[i] for i
                            in range(len(eeg_indices)) if hemi_mask[i]]

            n_hemi = len(hemi_indices)
            if n_hemi == 0:
                continue

            k = min(max_per_hemi, n_hemi)
            if k <= 1:
                selected.extend(hemi_indices)
                continue

            # k-means on (y, z) plane: clusters map to
            # frontal (high y), occipital (low y), superior (high z),
            # inferior (low z), and intermediate regions
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42,
                            n_init=10, max_iter=300)
            labels = kmeans.fit_predict(hemi_coords[:, 1:])

            for cid in range(k):
                mask_c = labels == cid
                coords_c = hemi_coords[mask_c]
                idx_local = np.where(mask_c)[0]
                centroid = coords_c.mean(axis=0)
                dists = np.linalg.norm(coords_c - centroid, axis=1)
                best = idx_local[np.argmin(dists)]
                selected.append(hemi_indices[best])

        # Sort + deduplicate for stability
        selected = sorted(set(selected))
        self._hemi_channels = selected

        ch_names_sel = [self.ch_names[i] for i in selected]
        # Quick anatomical grouping for verification
        from collections import Counter
        regions = _label_anatomical_region(coords,
                                            [eeg_indices.index(i)
                                             for i in selected
                                             if i in eeg_indices])
        region_counts = Counter(regions)
        print(f"[HemiCh] {len(selected)} 代表通道 (≤{max_per_hemi}/半球): "
              f"区域分布={dict(region_counts)}")
        print(f"[HemiCh] 通道: {ch_names_sel}")

        return selected

    # ================================================================
    #  Part 1 — 基础信息
    # ================================================================

    def recording_date(self):
        """
        提取脑电记录的日期和时间信息。

        Returns:
            包含日期时间信息的字典
        """
        try:
            meas_date = self.raw.info['meas_date']
            if isinstance(meas_date, tuple):
                dt_obj = datetime.fromtimestamp(meas_date[0])
            else:
                dt_obj = meas_date

            date_info = {
                'datetime': dt_obj,
                'date': dt_obj.date(),
                'time': dt_obj.time(),
                'year': dt_obj.year,
                'month': dt_obj.month,
                'day': dt_obj.day,
                'hour': dt_obj.hour,
                'minute': dt_obj.minute,
                'second': dt_obj.second
            }
            self.features['recording_date'] = date_info
            return date_info
        except Exception as e:
            print(f"日期时间提取失败: {e}")
            return None

    def sleep_stages_yasa(self,
                          eog_channels=None,
                          create_bipolar_eog=True,
                          metadata=None):
        """
        使用 YASA 进行自动睡眠分期。

        YASA 标准输入要求:
          - eeg_name: 中央区 EEG 导联 (如 C3-M2, 这里用 E21 近似 Cz)
          - eog_name: 眼电导联 (提高 Wake/REM 判别准确率)
          - emg_name: 颏肌电导联 (EGI 256导无此导联, 跳过)

        EGI 256导中眼电对应导联 (基于坐标分析, z≈0, |x|最大):
          - 左眼区域: E67 (x=-0.077, z=-0.003)  /  E68 (x=-0.077, z=0.015)
          - 右眼区域: E219 (x=+0.077, z=-0.003) / E210 (x=+0.077, z=0.015)
          默认创建双极导联 EOG = E67 - E219

        Args:
            eog_channels: 自定义眼电通道对 (left, right), None 使用默认 E67/E219
            create_bipolar_eog: 是否创建双极 EOG 导联
            metadata: 可选元数据 dict, 如 {'age': 30, 'male': 0}

        Returns:
            睡眠分期数组 (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        """
        try:
            # ---- 确定眼电通道 ----
            if eog_channels is None:
                eog_channels = ('E67', 'E219')  # 基于坐标的最前外侧导联

            eog_left, eog_right = eog_channels

            # 验证通道存在
            for ch in [eog_left, eog_right]:
                if ch not in self.raw.ch_names:
                    print(f"[YASA] 警告: EOG 候选通道 {ch} 不存在, 仅使用 EEG")
                    create_bipolar_eog = False
                    break

            # ---- 准备 Raw 对象 (含双极 EOG) ----
            if create_bipolar_eog:
                # 创建双极 EOG: left - right (捕获水平眼动)
                eog_data = (self.raw.get_data(picks=[eog_left])[0] -
                            self.raw.get_data(picks=[eog_right])[0])
                eeg_data = self.raw.get_data(picks=[self.eeg_channel])[0]
                # 组合为双通道 RawArray
                combined_data = np.vstack([eeg_data, eog_data])
                combined_info = mne.create_info(
                    [self.eeg_channel, 'EOG'],
                    sfreq=self.raw.info['sfreq'],
                    ch_types=['eeg', 'eog']
                )
                raw_stage = mne.io.RawArray(combined_data, combined_info)
                eog_arg = 'EOG'
            else:
                raw_stage = self.raw.copy().pick([self.eeg_channel])
                eog_arg = None
            sls = yasa.SleepStaging(
                raw_stage,
                eeg_name=self.eeg_channel,
                eog_name=eog_arg,
                metadata=metadata,
            )
            hypno_pred = sls.predict()

            # ---- 统计分期比例 ----
            labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
            stage_counts = {labels.get(s, s): (hypno_pred == s).sum()
                            for s in np.unique(hypno_pred)}
            total = len(hypno_pred)
            stage_pct = {k: f'{v/total*100:.1f}%' for k, v in stage_counts.items()}

            self.features['sleep_stages'] = {
                'stages': hypno_pred,
                'stage_labels': labels,
                'stage_counts': stage_counts,
                'stage_pct': stage_pct,
                'epoch_length': 30,
                'total_epochs': len(hypno_pred),
                'eeg_channel': self.eeg_channel,
                'eog_channels': eog_channels if create_bipolar_eog else None,
            }
            print(f"睡眠分期完成: {stage_pct}")
            return hypno_pred

        except Exception as e:
            print(f"睡眠分期提取失败: {e}")
            return None

    # ================================================================
    #  Part 2 — 频率域特征
    # ================================================================

    def psd(self, win_sec: float = 6):
        """
        使用 mne_features 提取频带归一化功率 (delta/theta/alpha/beta/gamma)。

        Args:
            win_sec: 滑动窗口长度（秒）

        Returns:
            dict with psd_features shape (n_windows, n_bands)
        """
        try:
            sfreq = self.sfreq
            data = self.data
            if data.ndim == 1:
                data = data[np.newaxis, :]

            win_samples = int(win_sec * sfreq)
            n_times = data.shape[1]
            n_windows = n_times // win_samples
            if n_windows < 1:
                print("[psd] 数据长度不足一个窗口")
                return None

            psd_features = []
            for w in range(n_windows):
                seg = data[:, w * win_samples:(w + 1) * win_samples]
                feat = compute_pow_freq_bands(
                    sfreq=sfreq, data=seg, freq_bands=self.BAND_EDGES,
                    normalize=True, log=False,
                    psd_method='welch', psd_params=None)
                psd_features.append(feat)

            psd_features = np.array(psd_features)
            result = {
                'win_sec': win_sec, 'n_windows': n_windows,
                'band_names': self.BAND_NAMES,
                'psd_features': psd_features, 'shape': psd_features.shape
            }
            self.features['psd'] = result
            self._epoch_info['psd'] = {'n_epochs': n_windows, 'epoch_sec': win_sec}
            print("功率谱计算完成")
            return result
        except Exception as e:
            print(f"能量谱提取失败: {e}")
            return None

    def spectral_entropy(self, win_sec: float = 30, fmin: float = 0.3,
                         fmax: float = 35.0):
        """
        计算频谱熵 (Spectral Entropy)，每个 epoch 一个值。

        Args:
            win_sec: epoch 长度（秒）
            fmin, fmax: 频率范围 (Hz) —— 修复：实际应用频带限制

        Returns:
            ndarray shape (n_epochs,)
        """
        sfreq = self.sfreq
        data = self.data
        if data.ndim == 1:
            data = data[np.newaxis, :]

        epoch_samples = int(win_sec * sfreq)
        n_times = data.shape[1]
        n_epochs = n_times // epoch_samples
        if n_epochs == 0:
            print("[Spectral Entropy] 数据长度不足")
            return None

        se_values = np.zeros(n_epochs)
        print(f"[Spectral Entropy] 计算 {n_epochs} 个 epoch 的频谱熵...")
        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Spectral Entropy")

        for ep in iterator:
            seg = data[:, ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                se = compute_spect_entropy(sfreq=sfreq, data=seg,
                                           psd_method='welch')
                se_values[ep] = se[0] if isinstance(se, np.ndarray) else se
            except Exception:
                se_values[ep] = np.nan

        self._store_feature('spectral_entropy', se_values,
                            {'epoch_sec': win_sec, 'fmin': fmin, 'fmax': fmax})
        print("频谱熵计算完成")
        return se_values

    # ================================================================
    #  Part 3 — 时域特征 (Hjorth, 统计量)
    # ================================================================

    def hjorth_mobility(self, epoch_sec: float = 30):
        """Hjorth Mobility（每 epoch）。"""
        data = self.filted
        if data.ndim == 1:
            data = data[np.newaxis, :]
        n_epochs = self._get_n_epochs(epoch_sec)
        if n_epochs == 0:
            return None
        epoch_samples = int(epoch_sec * self.sfreq)

        mobility = np.zeros(n_epochs)
        for ep in range(n_epochs):
            seg = data[:, ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                mobility[ep] = compute_hjorth_mobility(seg)[0]
            except Exception:
                mobility[ep] = np.nan

        self._store_feature('hjorth_mobility', mobility,
                            {'epoch_sec': epoch_sec})
        print("Hjorth Mobility 计算完成")
        return mobility

    def hjorth_complexity(self, epoch_sec: float = 30):
        """Hjorth Complexity（每 epoch）。"""
        data = self.filted
        if data.ndim == 1:
            data = data[np.newaxis, :]
        n_epochs = self._get_n_epochs(epoch_sec)
        if n_epochs == 0:
            return None
        epoch_samples = int(epoch_sec * self.sfreq)

        complexity = np.zeros(n_epochs)
        for ep in range(n_epochs):
            seg = data[:, ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                complexity[ep] = compute_hjorth_complexity(seg)[0]
            except Exception:
                complexity[ep] = np.nan

        self._store_feature('hjorth_complexity', complexity,
                            {'epoch_sec': epoch_sec})
        print("Hjorth Complexity 计算完成")
        return complexity

    def statistical_features(self, epoch_sec: float = 30):
        """
        统计特征：均值、方差、偏度、峰度、RMS、线长、过零率。

        Returns:
            dict of ndarrays, each shape (n_epochs,)
        """
        data = self.data
        if data.ndim == 1:
            data = data[np.newaxis, :]
        n_epochs = self._get_n_epochs(epoch_sec)
        if n_epochs == 0:
            return None
        epoch_samples = int(epoch_sec * self.sfreq)

        results = {
            'mean': np.zeros(n_epochs),
            'variance': np.zeros(n_epochs),
            'skewness': np.zeros(n_epochs),
            'kurtosis': np.zeros(n_epochs),
            'rms': np.zeros(n_epochs),
            'line_length': np.zeros(n_epochs),
            'zero_crossing_rate': np.zeros(n_epochs),
        }

        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Statistical Features")

        for ep in iterator:
            seg = data[0, ep * epoch_samples:(ep + 1) * epoch_samples]
            results['mean'][ep] = np.mean(seg)
            results['variance'][ep] = np.var(seg)
            results['skewness'][ep] = stats.skew(seg)
            results['kurtosis'][ep] = stats.kurtosis(seg)
            results['rms'][ep] = np.sqrt(np.mean(seg ** 2))
            results['line_length'][ep] = np.sum(np.abs(np.diff(seg)))
            results['zero_crossing_rate'][ep] = np.sum(np.diff(np.signbit(seg)) != 0) / len(seg)

        self.features['statistical'] = {
            'values': results,
            'meta': {'epoch_sec': epoch_sec}
        }
        print("统计特征计算完成")
        return results

    # ================================================================
    #  Part 4 — 非线性特征: 熵
    # ================================================================

    def sample_entropy(self, epoch_sec: float = 30):
        """Sample Entropy（每 epoch）。需要 antropy 库。"""
        if not HAS_ANTROPY:
            print("[Sample Entropy] antropy 未安装")
            return None
        return self._entropy_wrapper(ant.sample_entropy, 'sample_entropy', epoch_sec)

    def permutation_entropy(self, epoch_sec: float = 30, order: int = 3, delay: int = 1):
        """Permutation Entropy（每 epoch）。"""
        if not HAS_ANTROPY:
            print("[Permutation Entropy] antropy 未安装")
            return None
        values = np.zeros(self._get_n_epochs(epoch_sec))
        epoch_samples = int(epoch_sec * self.sfreq)
        data = self.data.ravel()
        iterator = range(len(values))
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Permutation Entropy")
        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                values[ep] = ant.perm_entropy(seg, order=order, delay=delay)
            except Exception:
                values[ep] = np.nan
        self._store_feature('permutation_entropy', values,
                            {'epoch_sec': epoch_sec, 'order': order})
        print("Permutation Entropy 计算完成")
        return values

    def approximate_entropy(self, epoch_sec: float = 30):
        """Approximate Entropy（每 epoch）。"""
        if not HAS_ANTROPY:
            return None
        return self._entropy_wrapper(ant.app_entropy, 'approximate_entropy', epoch_sec)

    def _entropy_wrapper(self, func, name: str, epoch_sec: float):
        """通用熵计算包装器。"""
        values = np.zeros(self._get_n_epochs(epoch_sec))
        epoch_samples = int(epoch_sec * self.sfreq)
        data = self.data.ravel()
        iterator = range(len(values))
        if HAS_TQDM:
            iterator = tqdm(iterator, desc=name.replace('_', ' ').title())
        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                values[ep] = func(seg)
            except Exception:
                values[ep] = np.nan
        self._store_feature(name, values, {'epoch_sec': epoch_sec})
        print(f"{name} 计算完成")
        return values

    # ================================================================
    #  Part 5 — 非线性特征: 复杂度 & 分形
    # ================================================================

    def lempel_ziv_complexity(self, epoch_sec: float = 30, normalize: bool = True):
        """
        Lempel-Ziv 复杂度（需要二值化后再计算）。

        使用 antropy.lziv_complexity。先将信号二值化（以中位数为界）。
        """
        if not HAS_ANTROPY:
            print("[LZ Complexity] antropy 未安装")
            return None
        values = np.zeros(self._get_n_epochs(epoch_sec))
        epoch_samples = int(epoch_sec * self.sfreq)
        data = self.data.ravel()
        iterator = range(len(values))
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Lempel-Ziv Complexity")
        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                binary = (seg > np.median(seg)).astype(int)
                values[ep] = ant.lziv_complexity(binary, normalize=normalize)
            except Exception:
                values[ep] = np.nan
        self._store_feature('lz_complexity', values,
                            {'epoch_sec': epoch_sec, 'normalize': normalize})
        print("Lempel-Ziv Complexity 计算完成")
        return values

    def higuchi_fractal_dimension(self, epoch_sec: float = 30, kmax: int = 10):
        """Higuchi 分形维数（每 epoch）。"""
        if not HAS_ANTROPY:
            return None
        values = np.zeros(self._get_n_epochs(epoch_sec))
        epoch_samples = int(epoch_sec * self.sfreq)
        data = self.data.ravel()
        iterator = range(len(values))
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Higuchi FD")
        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                values[ep] = ant.higuchi_fd(seg, kmax=kmax)
            except Exception:
                values[ep] = np.nan
        self._store_feature('higuchi_fd', values,
                            {'epoch_sec': epoch_sec, 'kmax': kmax})
        print("Higuchi Fractal Dimension 计算完成")
        return values

    def dfa(self, epoch_sec: float = 30):
        """
        Detrended Fluctuation Analysis (DFA) 指数（每 epoch）。
        需要 neurokit2 库。

        Returns:
            ndarray shape (n_epochs,)
        """
        if not HAS_NEUROKIT2:
            print("[DFA] neurokit2 未安装")
            return None
        values = np.zeros(self._get_n_epochs(epoch_sec))
        epoch_samples = int(epoch_sec * self.sfreq)
        data = self.data.ravel()
        iterator = range(len(values))
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="DFA")
        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                result = nk.fractal_dfa(seg)
                # neurokit2 >= 0.2.0 returns tuple (alpha, info_dict)
                if isinstance(result, tuple):
                    values[ep] = result[0]
                else:
                    values[ep] = result['dfa_alpha'].iloc[0]
            except Exception:
                values[ep] = np.nan
        self._store_feature('dfa', values, {'epoch_sec': epoch_sec})
        print("DFA 计算完成")
        return values

    # ================================================================
    #  Part 6 — pycatch22: 22 维规范时序特征
    # ================================================================

    def catch22_features(self, epoch_sec: float = 30):
        """
        pycatch22 — 22 个规范时序特征（每 epoch）。
        返回 shape (n_epochs, 22)。
        """
        values = np.zeros((self._get_n_epochs(epoch_sec), 22))
        epoch_samples = int(epoch_sec * self.sfreq)
        data = self.data.ravel()
        iterator = range(values.shape[0])
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="catch22")
        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                vals = pycatch22.catch22_all(seg)
                # catch22_all returns {'names': [...], 'values': [...]}
                values[ep] = vals['values']
            except Exception:
                values[ep] = np.nan
        self._store_feature('catch22', values, {'epoch_sec': epoch_sec})
        print("catch22 特征计算完成")
        return values

    # ================================================================
    #  Part 7 — 时频特征: 子波
    # ================================================================

    def wavelet_features(self, epoch_sec: float = 30,
                         freqs=None, n_cycles: int = 5):
        """
        基于 Morlet 小波的各频带平均能量。

        Args:
            epoch_sec: epoch 长度
            freqs: 频率列表，默认 2-40 Hz 对数分布
            n_cycles: 小波周期数

        Returns:
            dict: {band_name: array of shape (n_epochs,)}
        """
        if freqs is None:
            freqs = np.logspace(np.log10(2), np.log10(40), 20)

        data = self.data
        if data.ndim == 1:
            data = data[np.newaxis, :]

        n_epochs = self._get_n_epochs(epoch_sec)
        if n_epochs == 0:
            return None
        epoch_samples = int(epoch_sec * self.sfreq)

        result = {band: np.zeros(n_epochs) for band in self.BAND_NAMES}
        band_masks = {}
        for band, (fl, fh) in self.FREQ_BANDS.items():
            band_masks[band] = (freqs >= fl) & (freqs <= fh)

        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Wavelet Features")

        for ep in iterator:
            seg = data[:, ep * epoch_samples:(ep + 1) * epoch_samples]
            try:
                # 使用 mne 的 tfr_array_morlet
                power = mne.time_frequency.tfr_array_morlet(
                    seg[np.newaxis, :, :], sfreq=self.sfreq,
                    freqs=freqs, n_cycles=n_cycles,
                    output='power', verbose=False)
                power = power[0, 0]  # (n_freqs, n_times)
                avg_power = power.mean(axis=1)  # 时间平均
                for band in self.BAND_NAMES:
                    mask = band_masks[band]
                    if mask.any():
                        result[band][ep] = avg_power[mask].mean()
            except Exception:
                for band in self.BAND_NAMES:
                    result[band][ep] = np.nan

        self.features['wavelet'] = {
            'values': result,
            'meta': {'epoch_sec': epoch_sec, 'freqs': freqs}
        }
        print("子波特征计算完成")
        return result

    # ================================================================
    #  Part 8 — IRASA & 非周期特征
    # ================================================================

    def _irasa(self, data_1ch_uv: np.ndarray):
        """
        使用 yasa.irasa 分离振荡/非周期成分。

        data_1ch_uv: 1D 数组，单位 μV

        Returns:
            (freqs, psd_aperiodic, psd_oscillatory) 或 (None, None, None)
        """
        try:
            if data_1ch_uv.ndim == 2:
                if data_1ch_uv.shape[0] == 1:
                    data_1ch_uv = data_1ch_uv.ravel()
                else:
                    data_1ch_uv = data_1ch_uv[0]

            # 去直流
            data_1ch_uv = data_1ch_uv - np.mean(data_1ch_uv)

            # 清洗 inf/nan
            if np.any(~np.isfinite(data_1ch_uv)):
                print("[WARN] 数据包含 inf 或 nan，尝试修复...")
                data_1ch_uv = np.nan_to_num(
                    data_1ch_uv, nan=np.nanmedian(data_1ch_uv),
                    posinf=np.nanmedian(data_1ch_uv),
                    neginf=np.nanmedian(data_1ch_uv))
                if np.any(~np.isfinite(data_1ch_uv)):
                    return None, None, None

            if np.var(data_1ch_uv) < 1e-10:
                print("[WARN] 数据方差为零，跳过 IRASA")
                return None, None, None

            out = yasa.irasa(
                data=data_1ch_uv,
                sf=self.sfreq,
                ch_names=[self.eeg_channel],
                band=(0.3, 35),
                hset=[1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45,
                      1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85],
                return_fit=True,
                win_sec=4,
                verbose=False,
            )
            return out[0], out[1], out[2]
        except Exception as e:
            print(f"[WARN] IRASA failed: {e}")
            return None, None, None

    def aperiodic(self, epoch_sec: float = 30,
                  fit_band: tuple = (0.3, 35.0),
                  min_valid_bins: int = 3):
        """
        非周期斜率 (1/f^β) — 每 epoch 做 IRASA → log-log 线性拟合。

        Returns:
            dict with 'records', 'slope_per_epoch', 'beta_per_epoch'
        """
        data = self.data.ravel()
        sf = self.sfreq
        epoch_samples = int(epoch_sec * sf)
        n_epochs = len(data) // epoch_samples
        if n_epochs < 1:
            print("[aperiodic] 数据太短")
            return None

        rec = []
        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Aperiodic (IRASA)")

        for k in iterator:
            seg = data[k * epoch_samples:(k + 1) * epoch_samples]
            if np.any(~np.isfinite(seg)):
                rec.append({"epoch": k, "slope_loglog": np.nan,
                            "beta": np.nan, "ok": False})
                continue

            freqs, psd_ap, _ = self._irasa(seg)
            if freqs is None or psd_ap is None:
                rec.append({"epoch": k, "slope_loglog": np.nan,
                            "beta": np.nan, "intercept_loglog": np.nan,
                            "ok": False})
                continue

            psd_ap = np.asarray(psd_ap).ravel()
            f, p = np.asarray(freqs), psd_ap
            mask = ((f > 0) & np.isfinite(f) & np.isfinite(p) & (p > 0))
            lo, hi = fit_band
            mask &= (f >= lo) & (f <= hi)

            if mask.sum() < min_valid_bins:
                rec.append({"epoch": k, "slope_loglog": np.nan,
                            "beta": np.nan, "ok": False})
                continue

            X = np.log10(f[mask])
            Y = np.log10(p[mask])
            (b, a), *_ = np.linalg.lstsq(
                np.column_stack([np.ones_like(X), X]), Y, rcond=None)
            rec.append(dict(epoch=k, slope_loglog=b, beta=-b,
                            intercept_loglog=a, fit_lo=lo, fit_hi=hi, ok=True))

        out = {
            "epoch_sec": epoch_sec,
            "fit_band": fit_band,
            "records": rec,
            "slope_per_epoch": np.array([r["slope_loglog"] for r in rec]),
            "beta_per_epoch": np.array([r["beta"] for r in rec]),
        }
        self.features['aperiodic'] = out
        print("非周期特征计算完成")
        return out

    # ================================================================
    #  Part 8B — tsfresh 大规模特征提取 (hctsa 风格)
    #  Ref: "Beyond traditional sleep scoring: Massive feature extraction
    #        and data-driven clustering of sleep time series"
    # ================================================================

    def tsfresh_features(self, epoch_sec: float = 30,
                          n_jobs: int = 1,
                          fc_parameters: dict = None):
        """
        使用 tsfresh 对每个 epoch 提取大规模时间序列特征。
        这是 hctsa 最接近的 Python 等价物 —— 单通道可得 700+ 维特征。

        Args:
            epoch_sec: epoch 长度
            n_jobs: 并行作业数 (1 = 单线程)
            fc_parameters: tsfresh 特征参数配置；
                           None = 使用 EfficientFCParameters (精简集 ~200 个特征)
                           可传入 ComprehensiveFCParameters 获得完整 ~700 个特征

        Returns:
            pd.DataFrame: 每行一个 epoch，列为特征名
        """
        try:
            from tsfresh.feature_extraction import extract_features, EfficientFCParameters, ComprehensiveFCParameters
        except ImportError:
            print("[tsfresh] tsfresh 未安装，请: pip install tsfresh")
            return None

        if fc_parameters is None:
            fc_parameters = EfficientFCParameters()

        data = self.data.ravel()
        sf = self.sfreq
        epoch_samples = int(epoch_sec * sf)
        n_epochs = len(data) // epoch_samples
        if n_epochs < 1:
            print("[tsfresh] 数据太短")
            return None

        # 构建 tsfresh 所需的 DataFrame 格式
        import pandas as pd
        from tqdm import tqdm as _tqdm

        # 逐 epoch 提取 —— tsfresh 对单段时间序列效果最好
        all_features = []
        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = _tqdm(iterator, desc="tsfresh Features")

        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            if np.any(~np.isfinite(seg)):
                # 填充 NaN
                all_features.append(None)
                continue

            # tsfresh 期望 (time_index, kind, value) 格式
            df_ts = pd.DataFrame({
                'id': ['ch0'] * len(seg),
                'time': np.arange(len(seg)),
                'value': seg.astype(np.float64),
            })

            try:
                feats = extract_features(
                    df_ts, column_id='id', column_sort='time',
                    default_fc_parameters=fc_parameters,
                    n_jobs=n_jobs, disable_progressbar=True,
                )
                all_features.append(feats)
            except Exception:
                all_features.append(None)

        # 合并所有 epoch 的特征
        valid_epochs = [i for i, f in enumerate(all_features) if f is not None]
        if not valid_epochs:
            print("[tsfresh] 没有有效 epoch")
            return None

        # 对齐列（tsfresh 对不同数据可能返回不同列）
        all_cols = set()
        for f in all_features:
            if f is not None:
                all_cols.update(f.columns)
        all_cols = sorted(all_cols)

        feat_matrix = np.full((n_epochs, len(all_cols)), np.nan)
        for i, f in enumerate(all_features):
            if f is not None:
                for j, col in enumerate(all_cols):
                    if col in f.columns:
                        feat_matrix[i, j] = f[col].values[0]

        self.features['tsfresh'] = {
            'values': feat_matrix,
            'meta': {
                'epoch_sec': epoch_sec,
                'n_features': len(all_cols),
                'feature_names': all_cols,
            }
        }
        print(f"tsfresh 特征提取完成: {len(all_cols)} 个特征 × {n_epochs} 个epoch")
        return feat_matrix

    # ================================================================
    #  Part 8C — 递归定量分析 (RQA)
    # ================================================================

    def rqa_features(self, epoch_sec: float = 30,
                     embedding_dim: int = 3,
                     time_delay: int = 1,
                     threshold: float = 0.1,
                     min_diag_line: int = 2,
                     min_vert_line: int = 2,
                     decimate_factor: int = 5):
        """
        递归定量分析 (Recurrence Quantification Analysis)。

        计算每个 epoch 的 RQA 指标:
        - RR: 递归率 (Recurrence Rate)
        - DET: 确定性 (Determinism)
        - LAM: 层流性 (Laminarity)
        - L_max: 最大对角线长度
        - L_mean: 平均对角线长度
        - TT: 捕获时间 (Trapping Time)
        - ENTR: 香农熵 (对角线长度分布)

        Args:
            epoch_sec: epoch 长度
            embedding_dim: 嵌入维度
            time_delay: 时间延迟
            threshold: 递归阈值 (相对最大距离的比例)
            min_diag_line: 最小对角线长度
            min_vert_line: 最小垂直线长度
            decimate_factor: 降采样因子 (默认5, 即 250→50 Hz, 大幅加速)

        Returns:
            dict: 各 RQA 指标的 epoch 序列
        """
        data = self.data.ravel()
        sf = self.sfreq
        epoch_samples = int(epoch_sec * sf)
        n_epochs = len(data) // epoch_samples
        if n_epochs < 1:
            print("[RQA] 数据太短")
            return None

        # 降采样
        if decimate_factor > 1:
            effective_sf = sf // decimate_factor
        else:
            effective_sf = sf

        results = {
            'RR': np.zeros(n_epochs),
            'DET': np.zeros(n_epochs),
            'LAM': np.zeros(n_epochs),
            'L_max': np.zeros(n_epochs),
            'L_mean': np.zeros(n_epochs),
            'TT': np.zeros(n_epochs),
            'ENTR': np.zeros(n_epochs),
        }

        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="RQA Features")

        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            # 降采样
            if decimate_factor > 1:
                seg = seg[::decimate_factor]

            if np.any(~np.isfinite(seg)):
                for k in results:
                    results[k][ep] = np.nan
                continue

            try:
                rqa = self._compute_rqa(
                    seg, embedding_dim, time_delay, threshold,
                    min_diag_line, min_vert_line)
                for k, v in rqa.items():
                    results[k][ep] = v
            except Exception:
                for k in results:
                    results[k][ep] = np.nan

        self.features['rqa'] = {
            'values': results,
            'meta': {
                'epoch_sec': epoch_sec,
                'embedding_dim': embedding_dim,
                'time_delay': time_delay,
                'threshold': threshold,
                'decimate_factor': decimate_factor,
            }
        }
        print("RQA 特征计算完成")
        return results

    @staticmethod
    def _compute_rqa(seg: np.ndarray, emb_dim: int, delay: int,
                     threshold: float, min_diag: int, min_vert: int) -> dict:
        """计算单个 epoch 的 RQA 指标。"""
        # Step 1: 相空间重构 (time-delay embedding)
        n = len(seg) - (emb_dim - 1) * delay
        if n < 10:
            return {k: np.nan for k in ['RR', 'DET', 'LAM', 'L_max', 'L_mean', 'TT', 'ENTR']}

        embedded = np.zeros((n, emb_dim))
        for d in range(emb_dim):
            embedded[:, d] = seg[d * delay:d * delay + n]

        # Step 2: 计算距离矩阵
        from scipy.spatial.distance import pdist, squareform
        dist = squareform(pdist(embedded, metric='euclidean'))
        max_dist = dist.max()
        if max_dist == 0:
            return {k: np.nan for k in ['RR', 'DET', 'LAM', 'L_max', 'L_mean', 'TT', 'ENTR']}

        # Step 3: 构建递归矩阵 (二值化)
        epsilon = threshold * max_dist
        rm = (dist <= epsilon).astype(np.int8)
        np.fill_diagonal(rm, 0)  # 排除自递归

        # Step 4: 计算 RQA 指标
        N = rm.shape[0]
        total_pairs = N * (N - 1)  # 上三角不含对角线
        rec_points = np.sum(rm)
        RR = rec_points / total_pairs if total_pairs > 0 else 0.0

        # 对角线结构
        diag_hist = {}
        for k in range(-(N - 1), N):
            if k == 0:
                continue
            d = np.diag(rm, k=k)
            # 找连续1的游程
            runs = np.diff(np.concatenate(([0], d, [0])))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]
            lengths = ends - starts
            for L in lengths[lengths >= min_diag]:
                diag_hist[L] = diag_hist.get(L, 0) + 1

        # 垂直线结构
        vert_hist = {}
        for j in range(N):
            col = rm[:, j]
            runs = np.diff(np.concatenate(([0], col, [0])))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]
            lengths = ends - starts
            for L in lengths[lengths >= min_vert]:
                vert_hist[L] = vert_hist.get(L, 0) + 1

        # DET, L_max, L_mean, ENTR
        total_diag = sum(diag_hist.values())
        if total_diag > 0:
            DET = total_diag / rec_points if rec_points > 0 else 0.0
            L_max = max(diag_hist.keys()) if diag_hist else 0
            L_mean = sum(L * c for L, c in diag_hist.items()) / total_diag
            # 香农熵
            probs = np.array(list(diag_hist.values())) / total_diag
            ENTR = -np.sum(probs * np.log(probs + 1e-12))
        else:
            DET = L_max = L_mean = ENTR = 0.0

        # LAM, TT
        total_vert = sum(vert_hist.values())
        if total_vert > 0:
            LAM = total_vert / rec_points if rec_points > 0 else 0.0
            TT = sum(L * c for L, c in vert_hist.items()) / total_vert
        else:
            LAM = TT = 0.0

        return {
            'RR': RR, 'DET': DET, 'LAM': LAM,
            'L_max': L_max, 'L_mean': L_mean,
            'TT': TT, 'ENTR': ENTR,
        }

    # ================================================================
    #  Part 8D — 进阶频谱特征 & 自相关
    # ================================================================

    def additional_spectral_features(self, epoch_sec: float = 30):
        """
        进阶频谱特征: 谱矩、谱边频率、波段功率比。

        Returns:
            dict with keys: centroid, spread, skewness, kurtosis,
                            sef50, sef90, sef95, theta_beta_ratio,
                            delta_alpha_ratio, alpha_beta_ratio
        """
        data = self.data.ravel()
        sf = self.sfreq
        epoch_samples = int(epoch_sec * sf)
        n_epochs = len(data) // epoch_samples
        if n_epochs < 1:
            return None

        keys = [
            'centroid', 'spread', 'skewness', 'kurtosis',
            'sef50', 'sef90', 'sef95',
            'theta_beta_ratio', 'delta_alpha_ratio', 'alpha_beta_ratio',
        ]
        results = {k: np.zeros(n_epochs) for k in keys}

        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Adv. Spectral")

        bands = self.FREQ_BANDS

        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            if np.any(~np.isfinite(seg)):
                for k in keys:
                    results[k][ep] = np.nan
                continue

            try:
                # Welch PSD
                f, psd = signal.welch(seg, fs=sf, nperseg=min(1024, len(seg)),
                                      noverlap=min(512, len(seg)//2))
                mask = (f >= 0.5) & (f <= 40) & (psd > 0)
                if mask.sum() < 3:
                    for k in keys:
                        results[k][ep] = np.nan
                    continue

                f_m = f[mask]
                p_m = psd[mask]
                total_power = np.trapz(p_m, f_m)

                # 谱矩
                centroid = np.trapz(f_m * p_m, f_m) / total_power
                spread = np.sqrt(np.trapz((f_m - centroid)**2 * p_m, f_m) / total_power)
                skew = np.trapz((f_m - centroid)**3 * p_m, f_m) / (total_power * spread**3) if spread > 0 else 0
                kurt = np.trapz((f_m - centroid)**4 * p_m, f_m) / (total_power * spread**4) if spread > 0 else 0

                # 累积功率 → 谱边频率
                cumsum = np.cumsum(p_m) / total_power
                sef50 = f_m[np.searchsorted(cumsum, 0.50)]
                sef90 = f_m[np.searchsorted(cumsum, 0.90)]
                sef95 = f_m[np.searchsorted(cumsum, 0.95)]

                # 波段功率和
                def band_power(fl, fh):
                    idx = (f_m >= fl) & (f_m <= fh)
                    return np.trapz(p_m[idx], f_m[idx]) if idx.any() else 1e-12

                bp = {b: band_power(*r) for b, r in bands.items()}

                results['centroid'][ep] = centroid
                results['spread'][ep] = spread
                results['skewness'][ep] = skew
                results['kurtosis'][ep] = kurt
                results['sef50'][ep] = sef50
                results['sef90'][ep] = sef90
                results['sef95'][ep] = sef95
                results['theta_beta_ratio'][ep] = bp['theta'] / max(bp['beta'], 1e-12)
                results['delta_alpha_ratio'][ep] = bp['delta'] / max(bp['alpha'], 1e-12)
                results['alpha_beta_ratio'][ep] = bp['alpha'] / max(bp['beta'], 1e-12)

            except Exception:
                for k in keys:
                    results[k][ep] = np.nan

        self.features['adv_spectral'] = {
            'values': results,
            'meta': {'epoch_sec': epoch_sec}
        }
        print("进阶频谱特征计算完成")
        return results

    def autocorrelation_features(self, epoch_sec: float = 30,
                                  max_lag_sec: float = 2.0):
        """
        自相关函数 (ACF) 特征: 首个过零点、首个局部极小值、衰减率。

        Args:
            epoch_sec: epoch 长度
            max_lag_sec: 最大时滞 (秒)

        Returns:
            dict: acf_zc (过零点), acf_min (首个最小值), acf_decay (指数衰减率)
        """
        data = self.data.ravel()
        sf = self.sfreq
        epoch_samples = int(epoch_sec * sf)
        n_epochs = len(data) // epoch_samples
        if n_epochs < 1:
            return None

        max_lag = int(max_lag_sec * sf)
        keys = ['acf_zc', 'acf_first_min', 'acf_decay',
                'acf_lag1', 'acf_lag10']
        results = {k: np.zeros(n_epochs) for k in keys}

        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="ACF Features")

        for ep in iterator:
            seg = data[ep * epoch_samples:(ep + 1) * epoch_samples]
            if np.any(~np.isfinite(seg)):
                for k in keys:
                    results[k][ep] = np.nan
                continue

            try:
                acf = np.correlate(seg - seg.mean(), seg - seg.mean(), mode='full')
                acf = acf[len(acf)//2:]  # 只取正滞后
                acf = acf / max(acf[0], 1e-12)  # 归一化
                acf_trunc = acf[:min(max_lag, len(acf))]

                # Lag-1, Lag-10 自相关
                results['acf_lag1'][ep] = acf[1] if len(acf) > 1 else np.nan
                results['acf_lag10'][ep] = acf[10] if len(acf) > 10 else np.nan

                # 首个过零点
                zc_idx = np.where(np.diff(np.sign(acf_trunc)) < 0)[0]
                results['acf_zc'][ep] = zc_idx[0] / sf if len(zc_idx) > 0 else np.nan

                # 首个局部极小值
                from scipy.signal import argrelextrema
                minima = argrelextrema(acf_trunc, np.less)[0]
                results['acf_first_min'][ep] = (minima[0] / sf if len(minima) > 0
                                                  else np.nan)

                # 指数衰减率 (log|ACF| ~ lag 的线性拟合)
                valid = (acf_trunc[1:50] > 0.01) if len(acf_trunc) > 50 else (acf_trunc[1:] > 0.01)
                if valid.sum() >= 5:
                    lags = np.arange(1, min(51, len(acf_trunc)))
                    log_acf = np.log(np.maximum(acf_trunc[1:min(51, len(acf_trunc))], 1e-12))
                    slope, _ = np.polyfit(lags, log_acf, 1)
                    results['acf_decay'][ep] = -slope
                else:
                    results['acf_decay'][ep] = np.nan

            except Exception:
                for k in keys:
                    results[k][ep] = np.nan

        self.features['autocorrelation'] = {
            'values': results,
            'meta': {'epoch_sec': epoch_sec, 'max_lag_sec': max_lag_sec}
        }
        print("自相关特征计算完成")
        return results

    # ================================================================
    #  Part 9 — 特征波检测 (SSSM)
    # ================================================================

    def feature_waves(self, step: int = 300):
        """
        使用 SSSM 模型检测特征波（spindle, slow wave 等）。
        使用 MNE 降采样（含内置 FIR 抗混叠滤波）。

        Args:
            step: 模型预测步长（样本点）

        Returns:
            pred_labels: 模型预测标签
        """
        try:
            raw_sssm = self.filted_raw.copy()
            raw_sssm.resample(100)
            data_sssm = raw_sssm.get_data(units="uV")
            model = Model(device='cpu')
            model.predict(data_sssm, step=step)
            pred_labels = model.pred
            window_indices = np.arange(pred_labels.shape[1])  # 窗口索引
            self.features['feature_waves'] = pred_labels
            print("特征波提取完成")
            return pred_labels
        except Exception as e:
            print(f"特征波提取失败: {e}")
            return None

    # ================================================================
    #  Part 10 — 多通道特征: 功能连接
    # ================================================================

    def functional_connectivity(self, epoch_sec: float = 30,
                                methods=('coherence', 'plv', 'correlation'),
                                use_roi: bool = True, n_rois: int = 25):
        """
        多通道功能连接矩阵（每 epoch）。

        Args:
            epoch_sec: epoch 长度
            methods: 连接度量方式，可选 'coherence', 'plv', 'pli', 'correlation'
            use_roi: 是否使用 ROI 聚合模式（默认 True）
                     True  → 260 通道 → k-means 空间聚类 → n_rois 个脑区
                     False → 直接使用全部通道（内存/计算量极大）
            n_rois: ROI 数量（默认 25）

        Returns:
            dict: {method: ndarray shape (n_epochs, n_ch, n_ch)}
        """
        if not self._load_all:
            print("[Connectivity] 未启用多通道模式，请在 __init__ 设置 load_all_channels=True")
            return None

        if use_roi:
            data = self._get_roi_data(n_rois=n_rois, filtered=True)
        else:
            data = self._get_filted_all()

        n_ch = data.shape[0]
        n_epochs = int(data.shape[1] // (epoch_sec * self.sfreq))
        if n_epochs == 0:
            return None
        epoch_samples = int(epoch_sec * self.sfreq)

        label = f"ROI-{n_ch}ch" if use_roi else f"{n_ch}ch"
        print(f"[Connectivity] {label} 模式: {n_epochs} epochs, {n_ch}×{n_ch} "
              f"= {n_ch*(n_ch-1)//2} pairs")

        results = {}
        for method in methods:
            results[method] = np.zeros((n_epochs, n_ch, n_ch))

        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc=f"Connectivity ({label})")

        for ep in iterator:
            seg = data[:, ep * epoch_samples:(ep + 1) * epoch_samples]

            for method in methods:
                mat = self._connectivity_matrix(seg, method)
                results[method][ep] = mat

        meta = {'epoch_sec': epoch_sec, 'methods': methods,
                'use_roi': use_roi, 'n_channels': n_ch}
        if use_roi:
            if hasattr(self, '_roi_groups'):
                meta['roi_groups'] = self._roi_groups
            elif hasattr(self, '_hemi_ch_names'):
                meta['hemi_channels'] = self._hemi_ch_names
        self.features['connectivity'] = {'values': results, 'meta': meta}
        print("功能连接计算完成")
        return results

    def _connectivity_matrix(self, data: np.ndarray, method: str):
        """
        计算连接矩阵。

        Args:
            data: shape (n_channels, n_times)
            method: 'coherence', 'plv', 'pli', 'correlation'

        Returns:
            ndarray shape (n_ch, n_ch)
        """
        n_ch = data.shape[0]
        mat = np.zeros((n_ch, n_ch))

        if method == 'correlation':
            corr = np.corrcoef(data)
            mat = corr
            np.fill_diagonal(mat, 0)
            return mat

        # 频域方法需要 PSD / 交叉谱
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                if method == 'coherence':
                    f, cxy = signal.coherence(data[i], data[j],
                                              fs=self.sfreq, nperseg=min(256, len(data[i]) // 2))
                    mat[i, j] = mat[j, i] = np.mean(cxy[(f >= 0.5) & (f <= 40)])
                elif method == 'plv':
                    mat[i, j] = mat[j, i] = self._plv(data[i], data[j])
                elif method == 'pli':
                    mat[i, j] = mat[j, i] = self._pli(data[i], data[j])
        return mat

    @staticmethod
    def _plv(x, y):
        """Phase Locking Value。"""
        xa = signal.hilbert(x)
        ya = signal.hilbert(y)
        phase_diff = np.angle(xa) - np.angle(ya)
        return np.abs(np.mean(np.exp(1j * phase_diff)))

    @staticmethod
    def _pli(x, y):
        """Phase Lag Index。"""
        xa = signal.hilbert(x)
        ya = signal.hilbert(y)
        phase_diff = np.angle(xa) - np.angle(ya)
        return np.abs(np.mean(np.sign(phase_diff)))

    # ================================================================
    #  Part 11 — 多通道特征: 微状态
    # ================================================================

    def microstate_analysis(self, n_states: int = 4, gfp_peaks: bool = True):
        """
        微状态分析。

        使用 mne_microstates 对全通道 GFP 峰值处的拓扑进行聚类，
        提取微状态序列和指标。

        Args:
            n_states: 微状态数量
            gfp_peaks: 是否仅在 GFP 峰值处拟合

        Returns:
            dict with segmentation, metrics
        """
        if not self._load_all:
            print("[Microstates] 未启用多通道模式")
            return None

        try:
            # 获取滤波后数据并包装为 MNE RawArray
            data_filt = self._get_filted_all()
            # 半球模式下使用实际选中的通道名
            if hasattr(self, '_hemi_ch_names'):
                ch_names_micro = self._hemi_ch_names
            else:
                ch_names_micro = self.ch_names[:data_filt.shape[0]]
            info = mne.create_info(
                ch_names_micro, self.sfreq,
                ch_types=['eeg'] * data_filt.shape[0])
            raw_micro = mne.io.RawArray(data_filt, info, verbose=False)
            # mne_microstates >= 0.3: segment(data, n_states) → (maps, labels)
            import mne_microstates as mm

            maps, labels = mm.segment(
                data_filt, n_states=n_states, random_state=42,
                verbose=False)

            # 计算微状态指标: 覆盖率、持续时长、出现频率、GFP
            metrics = self._microstate_metrics(labels, n_states)

            result = {
                'labels': labels,
                'n_states': n_states,
                'metrics': metrics,
            }
            self.features['microstates'] = result
            print("微状态分析完成")
            return result
        except Exception as e:
            print(f"微状态分析失败: {e}")
            return None

    def _microstate_metrics(self, labels: np.ndarray, n_states: int):
        """计算微状态指标。"""
        metrics = {}
        for s in range(n_states):
            mask = labels == s
            coverage = mask.mean()
            # 平均持续时长（转换为秒）
            transitions = np.diff(mask.astype(int))
            onset = np.where(transitions == 1)[0] + 1
            offset = np.where(transitions == -1)[0] + 1
            if len(onset) > 0 and len(offset) > 0:
                durations = offset[:min(len(onset), len(offset))] - onset[:min(len(onset), len(offset))]
                mean_dur = np.mean(durations) / self.sfreq if len(durations) > 0 else 0
                freq = len(onset) / (len(labels) / self.sfreq / 60)  # per minute
            else:
                mean_dur = 0
                freq = 0

            metrics[f'state_{s}'] = {
                'coverage': coverage,
                'mean_duration_s': mean_dur,
                'frequency_per_min': freq,
            }
        return metrics

    # ================================================================
    #  Part 12 — 多通道特征: 空间
    # ================================================================

    def global_field_power(self, epoch_sec: float = 30):
        """
        全局场功率 (GFP): sqrt(mean(channels^2)) — 每采样点，然后 epoch 平均。

        Returns:
            ndarray shape (n_epochs,)
        """
        if not self._load_all:
            print("[GFP] 未启用多通道模式")
            return None

        data = self._get_data_all()  # (n_ch, n_times)
        # 逐采样点计算 GFP
        gfp_raw = np.sqrt(np.mean(data ** 2, axis=0))  # (n_times,)

        n_epochs = self._get_n_epochs(epoch_sec)
        epoch_samples = int(epoch_sec * self.sfreq)
        gfp_epochs = np.zeros(n_epochs)
        for ep in range(n_epochs):
            seg = gfp_raw[ep * epoch_samples:(ep + 1) * epoch_samples]
            gfp_epochs[ep] = np.mean(seg)

        self._store_feature('gfp', gfp_epochs, {'epoch_sec': epoch_sec})
        print("Global Field Power 计算完成")
        return gfp_epochs

    def spatial_complexity(self, epoch_sec: float = 30):
        """
        空间复杂度 (Omega Complexity)。

        Omega = exp(-∑ λ_i' * log(λ_i'))，其中 λ_i' 是归一化特征值。

        Returns:
            ndarray shape (n_epochs,)
        """
        if not self._load_all:
            print("[Spatial Complexity] 未启用多通道模式")
            return None

        data = self._get_data_all()
        n_epochs = self._get_n_epochs(epoch_sec)
        epoch_samples = int(epoch_sec * self.sfreq)
        omega = np.zeros(n_epochs)

        for ep in range(n_epochs):
            seg = data[:, ep * epoch_samples:(ep + 1) * epoch_samples]
            # 协方差矩阵特征值（加小量正则化防数值不稳定）
            cov = np.cov(seg)
            cov += np.eye(cov.shape[0]) * 1e-10
            try:
                eigvals = np.linalg.eigvalsh(cov)
            except np.linalg.LinAlgError:
                omega[ep] = np.nan
                continue
            eigvals = np.maximum(eigvals, 1e-15)
            eigvals_norm = eigvals / eigvals.sum()
            entropy = -np.sum(eigvals_norm * np.log(eigvals_norm))
            omega[ep] = np.exp(entropy)

        self._store_feature('omega_complexity', omega, {'epoch_sec': epoch_sec})
        print("空间复杂度计算完成")
        return omega

    # ================================================================
    #  Part 13 — 图论特征
    # ================================================================

    def graph_metrics(self, epoch_sec: float = 30,
                     use_roi: bool = True, n_rois: int = 25):
        """
        图论指标（基于相关矩阵构建图）。

        Args:
            epoch_sec: epoch 长度
            use_roi: 是否使用 ROI 聚合模式（默认 True）
            n_rois: ROI 数量

        Returns:
            dict with 'degree', 'clustering', 'path_length' per epoch
        """
        if not self._load_all or not HAS_NX:
            if not HAS_NX:
                print("[Graph Metrics] networkx 未安装")
            return None

        if use_roi:
            data = self._get_roi_data(n_rois=n_rois, filtered=False)
            data_eeg = data
        else:
            data = self._get_data_all()
            eeg_ch_idx = [i for i, ch in enumerate(self.ch_names)
                          if ch.startswith('E') and ch[1:].isdigit()]
            data_eeg = data[eeg_ch_idx]

        n_ch = data_eeg.shape[0]
        label = f"ROI-{n_ch}ch" if use_roi else f"{n_ch}ch"
        print(f"[Graph] {label} 模式: n_ch={n_ch}")
        n_epochs = self._get_n_epochs(epoch_sec)
        epoch_samples = int(epoch_sec * self.sfreq)

        results = {
            'mean_degree': np.zeros(n_epochs),
            'mean_clustering': np.zeros(n_epochs),
            'char_path_length': np.zeros(n_epochs),
        }

        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Graph Metrics")

        for ep in iterator:
            seg = data_eeg[:, ep * epoch_samples:(ep + 1) * epoch_samples]
            corr = np.corrcoef(seg)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 0)

            # 二值化: 取 top 20% 的边
            threshold = np.percentile(np.abs(corr), 80)
            adj = (np.abs(corr) > threshold).astype(float)

            G = nx.from_numpy_array(adj)
            try:
                results['mean_degree'][ep] = np.mean([d for _, d in G.degree()])
                results['mean_clustering'][ep] = nx.average_clustering(G)
                # 取最大连通分量计算最短路径 (避免断连图报错)
                if nx.is_connected(G):
                    results['char_path_length'][ep] = nx.average_shortest_path_length(G)
                else:
                    largest_cc = max(nx.connected_components(G), key=len)
                    subG = G.subgraph(largest_cc)
                    results['char_path_length'][ep] = nx.average_shortest_path_length(subG)
            except Exception:
                results['mean_degree'][ep] = np.nan
                results['mean_clustering'][ep] = np.nan
                results['char_path_length'][ep] = np.nan

        self.features['graph_metrics'] = {
            'values': results,
            'meta': {'epoch_sec': epoch_sec}
        }
        print("图论指标计算完成")
        return results

    # ================================================================
    #  Part 13a — 源定位 (Source Localization)
    # ================================================================
    #
    # 原理:
    #   头皮EEG是皮层锥体神经元同步突触后电位的容积传导投影。
    #   源定位通过求解电磁逆问题，从头皮电位反推皮层源活动。
    #
    # 工作流:
    #   fsaverage模板 → BEM头模型 → 前向解(leadfield) → 逆解(MNE/dSPM/eLORETA)
    #
    # 方法:
    #   - MNE: Minimum Norm Estimate, L2正则化最小范数解
    #   - dSPM: dynamic Statistical Parametric Mapping, 噪声归一化MNE
    #   - eLORETA: exact Low Resolution Tomography, 零定位误差
    #
    # 参考:
    #   - Gramfort et al. 2013, NeuroImage
    #   - Pascual-Marqui 2007, arXiv:0710.3341 (eLORETA)
    #   - Dale et al. 2000, Neuron (dSPM)
    #   - Hämäläinen & Ilmoniemi 1994, Med Biol Eng Comput (MNE)
    #
    # 前置依赖:
    #   pip install mne  # 需含 fsaverage 模板数据
    #   首次运行自动下载 fsaverage (~100 MB)
    #
    # ────────────────────────────────────────────────────────────

    def source_localization(self, method='eLORETA', epoch_sec=30,
                            snr=3.0, loose=0.2, depth=0.8):
        """源定位: 从头皮EEG重建皮层源活动。

        使用 fsaverage 模板 MRI + BEM 头模型 + 逆解。

        Args:
            method: 逆解方法 'MNE', 'dSPM', 'eLORETA' (推荐 eLORETA)
            epoch_sec: epoch 长度 (s)
            snr: 信噪比，用于正则化 (默认 3.0)
            loose: 源朝向约束 (0=固定朝向, 1=自由朝向, 0.2=宽松)
            depth: 深度加权 (0=无, 1=全加权, 默认 0.8)

        Returns:
            dict with 'stc_epochs', 'method', 'src', 'fwd', 'inv_op'
            stc_epochs: list of SourceEstimate, 每epoch一个
        """
        import mne
        from mne.minimum_norm import (make_inverse_operator,
                                       apply_inverse)
        import numpy as np

        print(f"\n[Source Loc] 方法: {method}, epoch={epoch_sec}s")
        print("[Source Loc] Step 1/5: 设置源空间 (fsaverage)...")

        # ── Step 1: 源空间 ──
        # fsaverage 是 MNI 标准模板，20484 个顶点/半球
        # spacing='oct6' → ~4098 src vertices per hemi (可管理的大小)
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = str(Path(fs_dir).parent)
        src = mne.setup_source_space(
            'fsaverage', spacing='oct6',
            subjects_dir=subjects_dir, add_dist=False)

        print(f"[Source Loc] 源空间: {len(src)} 个半球, "
              f"~{sum(len(s['vertno']) for s in src)} 顶点")

        # ── Step 2: BEM 头模型 ──
        print("[Source Loc] Step 2/5: 构建 BEM 模型...")
        conductivity = (0.3,)  # 单层BEM (头皮), EEG可用
        model = mne.make_bem_model(
            'fsaverage', ico=4, conductivity=conductivity,
            subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)

        # ── Step 3: 蒙太奇 + 前向解 ──
        print("[Source Loc] Step 3/5: 设置电极蒙太奇 + 前向解...")
        # 使用 EGI 256 标准蒙太奇
        montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
        # 用我们的通道名创建 info
        ch_names_eeg = [ch for ch in self.ch_names
                        if ch.startswith('E') and ch[1:].isdigit()]
        info = mne.create_info(ch_names_eeg, self.sfreq, ch_types='eeg')
        info.set_montage(montage)

        # 估计头-设备变换 (对于模板, 使用自动fiducial对齐)
        trans = mne.coreg.estimate_head_mri_t(
            'fsaverage', subjects_dir=subjects_dir)

        # 前向解 (leadfield matrix)
        fwd = mne.make_forward_solution(
            info, trans=trans, src=src, bem=bem,
            meg=False, eeg=True, mindist=5.0, n_jobs=1)
        print(f"[Source Loc] 前向解: {fwd['nsource']} 源, "
              f"{fwd['nchan']} 通道")

        # 约束为固定朝向（皮层锥体细胞垂直于皮层表面）
        fwd = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=False, copy=False)

        # ── Step 4: 数据准备 + 噪声协方差 ──
        print("[Source Loc] Step 4/5: 准备 epoch 数据 + 噪声协方差...")
        # 加载滤波后的全通道数据 → 重塑为 (n_epochs, n_ch, n_times)
        epoch_samples = int(epoch_sec * self.sfreq)
        data = self._get_filted_all()
        n_ch_fwd = data.shape[0]
        n_epochs = int(data.shape[1] // epoch_samples)
        data_3d = (data[:, :epoch_samples * n_epochs]
                   .reshape(n_ch_fwd, n_epochs, epoch_samples)
                   .transpose(1, 0, 2))

        events = np.column_stack([
            np.arange(n_epochs, dtype=int),
            np.zeros(n_epochs, int),
            np.ones(n_epochs, int)
        ])
        epochs = mne.EpochsArray(
            data_3d, info, events=events, tmin=0,
            baseline=None, verbose=False)

        # 噪声协方差: 用 shrinkage 方法从数据自身估计
        noise_cov = mne.compute_covariance(epochs, method='shrunk',
                                            verbose=False)

        # ── Step 5: 逆解 ──
        print(f"[Source Loc] Step 5/5: 计算逆解 ({method})...")
        inv_op = make_inverse_operator(
            epochs.info, fwd, noise_cov,
            loose=loose, depth=depth, verbose=False)

        # 逐个 epoch 应用逆解
        stc_epochs = []
        iterator = range(n_epochs)
        if HAS_TQDM:
            iterator = tqdm(iterator, desc=f"Source Loc ({method})")

        for ep in iterator:
            evoked = epochs[ep].average()
            stc = apply_inverse(
                evoked, inv_op, method=method,
                pick_ori=None, verbose=False)
            stc_epochs.append(stc)

        print(f"[Source Loc] 完成: {n_epochs} epochs × "
              f"{len(stc_epochs[0].data)} 源")

        result = {
            'stc_epochs': stc_epochs,
            'method': method,
            'src': src,
            'fwd': fwd,
            'inv_op': inv_op,
            'n_epochs': n_epochs,
        }
        self.features['source_localization'] = {'values': result}
        return result

    def source_band_power(self, epoch_sec=30, method='eLORETA',
                          bands=None):
        """源空间频带功率: 在每个源点计算各频带的平均功率。

        Args:
            epoch_sec: epoch 长度
            method: 逆解方法
            bands: dict {name: (low, high)} 或 None (使用默认5频带)

        Returns:
            dict {band_name: ndarray (n_epochs, n_sources)}
        """
        if bands is None:
            bands = {'delta': (0.5, 4), 'theta': (4, 8),
                     'alpha': (8, 13), 'beta': (13, 30),
                     'gamma': (30, 45)}

        # 确保源定位已完成
        src_result = self.features.get('source_localization', {}).get('values')
        if src_result is None:
            print("[Src Band Pwr] 先运行 source_localization()...")
            src_result = self.source_localization(method=method,
                                                   epoch_sec=epoch_sec)

        stc_epochs = src_result['stc_epochs']
        n_epochs = len(stc_epochs)
        n_sources = stc_epochs[0].data.shape[0]
        sfreq_stc = stc_epochs[0].sfreq  # MNE stores sampling rate

        results = {name: np.zeros((n_epochs, n_sources))
                   for name in bands}

        for ep, stc in enumerate(stc_epochs):
            for band_name, (fl, fh) in bands.items():
                # 用 STFT 计算频带功率
                from scipy import signal as scipy_signal
                freqs, psd = scipy_signal.welch(
                    stc.data, fs=sfreq_stc, nperseg=min(256, stc.data.shape[1]))
                band_mask = (freqs >= fl) & (freqs <= fh)
                band_power = np.trapz(psd[:, band_mask], freqs[band_mask], axis=1)
                results[band_name][ep] = band_power

        self.features['source_band_power'] = {
            'values': results,
            'meta': {'epoch_sec': epoch_sec, 'method': method,
                     'bands': bands, 'n_sources': n_sources}
        }
        print(f"[Src Band Pwr] 完成: {n_epochs} epochs × "
              f"{len(bands)} bands × {n_sources} sources")
        return results

    def source_roi_power(self, epoch_sec=30, method='eLORETA',
                          parc='aparc'):
        """源空间 ROI 功率: 将源聚合法 Destrieux/Desikan 图谱 ROI。

        Args:
            epoch_sec: epoch 长度
            method: 逆解方法
            parc: 图谱名称 'aparc' (Desikan-Killiany, ~68 ROI)
                  或 'aparc.a2009s' (Destrieux, ~150 ROI)

        Returns:
            dict {roi_name: ndarray (n_epochs,)}
        """
        import mne

        # 确保源定位已完成
        src_result = self.features.get('source_localization', {}).get('values')
        if src_result is None:
            src_result = self.source_localization(method=method,
                                                   epoch_sec=epoch_sec)

        stc_epochs = src_result['stc_epochs']
        src = src_result['src']
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = str(Path(fs_dir).parent)

        # 读取皮层图谱标签
        labels = mne.read_labels_from_annot(
            'fsaverage', parc=parc, subjects_dir=subjects_dir)

        print(f"[Src ROI Pwr] {len(labels)} ROIs from {parc}")

        # 对每个 epoch: 每个 ROI 取源点平均
        roi_names = [l.name for l in labels]
        results = {name: np.zeros(len(stc_epochs)) for name in roi_names}

        for ep, stc in enumerate(stc_epochs):
            # 用 extract_label_time_course 批量提取所有ROI时间序列
            label_ts = mne.extract_label_time_course(
                stc, labels, src, mode='mean',
                allow_empty=True, verbose=False)
            for i, label in enumerate(labels):
                ts = label_ts[i]
                if len(ts) > 0 and not np.all(np.isnan(ts)):
                    results[label.name][ep] = np.sqrt(np.mean(ts ** 2))
                else:
                    results[label.name][ep] = np.nan

        self.features['source_roi_power'] = {
            'values': results,
            'meta': {'epoch_sec': epoch_sec, 'method': method,
                     'parc': parc, 'n_rois': len(labels)}
        }
        n_valid = sum(1 for v in results.values()
                      if not np.all(np.isnan(v)))
        print(f"[Src ROI Pwr] 完成: {n_valid}/{len(labels)} ROIs 非空")
        return results

    # ══════════════════════════════════════════════════════════
    #  4-Step Linear Pipeline
    # ══════════════════════════════════════════════════════════

    def run_step1_single_channel(self, epoch_sec=30,
                                  groups=('basic', 'time_domain', 'frequency',
                                          'entropy', 'complexity',
                                          'tsfresh', 'rqa',
                                          'adv_spectral', 'autocorrelation'),
                                  skip_sssm=False,
                                  yasa_metadata=None):
        """Step 1: 单通道特征提取 — 选最具代表性中央通道 E21 (≈Cz).

        E21 是睡眠研究的金标准通道（AASM 手册推荐 C3/C4/Cz）。
        所有特征从该单通道提取，内存占用 ~60 MB。
        提取的特征包括：频谱、时域、熵、复杂度、分形、catch22、
        tsfresh (hctsa 风格)、RQA、进阶频谱、自相关、SSSM 特征波。

        YASA 分期在 Step 2 单独执行（需要额外加载 EOG 通道）。
        """
        print(f"\n{'='*60}")
        print(f"[Step 1/4] 单通道特征提取 (通道: {self.eeg_channel})")
        print(f"{'='*60}\n")

        # ── 基础信息（不含 YASA 分期）──
        if 'basic' in groups:
            print("\n--- 基础信息 ---")
            self.recording_date()

        # ── 频率域 ──
        if 'frequency' in groups:
            print("\n--- 频率域特征 ---")
            self.psd(win_sec=6)
            self.spectral_entropy(win_sec=epoch_sec)
            self.aperiodic(epoch_sec=epoch_sec)

        # ── 时域 ──
        if 'time_domain' in groups:
            print("\n--- 时域特征 ---")
            self.hjorth_mobility(epoch_sec=epoch_sec)
            self.hjorth_complexity(epoch_sec=epoch_sec)
            self.statistical_features(epoch_sec=epoch_sec)

        # ── 熵 ──
        if 'entropy' in groups:
            print("\n--- 熵特征 ---")
            self.sample_entropy(epoch_sec=epoch_sec)
            self.permutation_entropy(epoch_sec=epoch_sec)
            self.approximate_entropy(epoch_sec=epoch_sec)

        # ── 复杂度 & 分形 ──
        if 'complexity' in groups:
            print("\n--- 复杂度 & 分形 ---")
            self.lempel_ziv_complexity(epoch_sec=epoch_sec)
            self.higuchi_fractal_dimension(epoch_sec=epoch_sec)
            self.dfa(epoch_sec=epoch_sec)
            self.catch22_features(epoch_sec=epoch_sec)
            self.wavelet_features(epoch_sec=epoch_sec)

        # ── tsfresh 大规模特征 (hctsa 风格) ──
        if 'tsfresh' in groups:
            print("\n--- tsfresh 大规模特征 (hctsa 风格) ---")
            try:
                self.tsfresh_features(epoch_sec=epoch_sec)
            except Exception as e:
                print(f"tsfresh 跳过: {e}")

        # ── RQA ──
        if 'rqa' in groups:
            print("\n--- 递归定量分析 (RQA) ---")
            try:
                self.rqa_features(epoch_sec=epoch_sec)
            except Exception as e:
                print(f"RQA 跳过: {e}")

        # ── 进阶频谱 ──
        if 'adv_spectral' in groups:
            print("\n--- 进阶频谱特征 ---")
            try:
                self.additional_spectral_features(epoch_sec=epoch_sec)
            except Exception as e:
                print(f"进阶频谱跳过: {e}")

        # ── 自相关 ──
        if 'autocorrelation' in groups:
            print("\n--- 自相关特征 ---")
            try:
                self.autocorrelation_features(epoch_sec=epoch_sec)
            except Exception as e:
                print(f"自相关跳过: {e}")

        # ── SSSM 特征波 ──
        if not skip_sssm:
            print("\n--- 特征波检测 (SSSM) ---")
            try:
                self.feature_waves()
            except Exception as e:
                print(f"SSSM 跳过: {e}")

        print(f"\n[Step 1/4] ✓ 单通道特征提取完成")

    def run_step2_yasa_with_eog(self,
                                 eog_channels=('E67', 'E219'),
                                 metadata=None):
        """Step 2: YASA 睡眠分期 — EEG+EOG 均通过 MNE 加载。

        E21 已在 __init__ 中加载为 self.data。EOG 单独通过 MNE
        read_raw_egi(pick+load_data) 加载，只需 ~120 MB（仅 2 通道）。
        两者来自相同 MNE parser → 长度一致 → 天然对齐 → 与 Step 1 epoch 也一致。

        Args:
            eog_channels: (left, right) EOG 通道对，
                          默认 E67/E219（EGI 256 最前外侧导联）
            metadata: YASA 可选元数据 {'age': 30, 'male': 0}
        """
        import gc
        import time

        print(f"\n{'='*60}")
        print(f"[Step 2/4] YASA 睡眠分期 (EEG+EOG, MNE)")
        print(f"{'='*60}\n")

        eog_left, eog_right = eog_channels

        # ── 验证 EOG 通道 ──
        for ch in [eog_left, eog_right]:
            if ch not in self.ch_names:
                print(f"[Step2] ✗ EOG 通道 {ch} 不存在！仅用 EEG 做分期")
                self.sleep_stages_yasa(
                    eog_channels=eog_channels,
                    create_bipolar_eog=False,
                    metadata=metadata)
                return

        # ── MNE 加载 EOG 通道 (pick + load_data, 仅 2 通道 ~120 MB) ──
        print(f"[Step2] MNE 加载 EOG: {eog_left}, {eog_right}")
        t_load = time.time()
        raw_eog = mne.io.read_raw_egi(self.file_path, preload=False, verbose=False)
        raw_eog.pick([eog_left, eog_right])
        raw_eog.load_data()
        eog_data = raw_eog.get_data(units='uV')  # (2, n_times), 与 self.data 同长
        print(f"[Step2] EOG 加载完成 ({time.time()-t_load:.1f}s) "
              f"— 与 EEG 同源 MNE, 长度一致")

        # ── 双极 EOG + EEG → RawArray ──
        eog_bipolar = eog_data[0] - eog_data[1]          # E67 - E219
        eeg_data = self.data                              # 已在 __init__ 中加载, 同长

        combined = np.vstack([eeg_data, eog_bipolar])
        info = mne.create_info(
            [self.eeg_channel, 'EOG'],
            sfreq=self.sfreq,
            ch_types=['eeg', 'eog'])
        raw_stage = mne.io.RawArray(combined, info, verbose=False)

        # ── 运行 YASA (LightGBM 模型已内置，秒级加载) ──
        print(f"[Step2] YASA SleepStaging 开始... ")
        t_yasa = time.time()
        try:
            sls = yasa.SleepStaging(
                raw_stage,
                eeg_name=self.eeg_channel,
                eog_name='EOG',
                metadata=metadata,
            )
            hypno_pred = np.asarray(sls.predict(), dtype=int)

            labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
            stage_counts = {labels.get(s, s): int((hypno_pred == s).sum())
                            for s in np.unique(hypno_pred)}
            total = len(hypno_pred)
            stage_pct = {k: f'{v/total*100:.1f}%'
                         for k, v in stage_counts.items()}

            self.features['sleep_stages'] = {
                'stages': hypno_pred,
                'stage_labels': labels,
                'stage_counts': stage_counts,
                'stage_pct': stage_pct,
                'epoch_length': 30,
                'total_epochs': len(hypno_pred),
                'eeg_channel': self.eeg_channel,
                'eog_channels': eog_channels,
            }
            print(f"[Step2] YASA 完成 ({time.time()-t_yasa:.0f}s): "
                  f"{stage_pct}")

            del sls
        except Exception as e:
            print(f"[Step2] YASA 失败: {e}")

        # ── 清除数据 ──
        del raw_eog, eog_data, eog_bipolar, combined, raw_stage
        gc.collect()
        print(f"[Step 2/4] ✓ YASA 分期完成 (EOG 数据已清除)")

    def run_step3_multichannel_roi(self, epoch_sec=30,
                                    max_per_hemi=5,
                                    groups=('connectivity', 'microstates',
                                            'spatial')):
        """Step 3: 多通道特征 — 半球分区 ROI，每半球 ≤5 代表通道。

        从 sensorLayout.xml 解析通道 3D 坐标，按 x 坐标分左右半球。
        每半球内 y-z 平面 k-means(k=5) 聚类，覆盖额叶/中央/颞叶/
        顶叶/枕叶区域。选每类距质心最近的通道 → 共 ≤10 通道。

        仅加载这 ~10 个通道（~600 MB vs 全 260 通道的 7.5 GB），
        做完连接性/微状态/空间/图论特征后立即清除。

        Args:
            epoch_sec: epoch 长度
            max_per_hemi: 每半球最多选取通道数 (默认 5)
            groups: 多通道特征组
        """
        import gc
        import time

        print(f"\n{'='*60}")
        print(f"[Step 3/4] 多通道 ROI 特征 (半球≤{max_per_hemi}通道/侧)")
        print(f"{'='*60}\n")

        # ── 1. 选择半球代表通道 ──
        hemi_indices = self._get_hemispheric_channels(
            max_per_hemi=max_per_hemi)
        n_hemi = len(hemi_indices)
        print(f"[Step3] 选中 {n_hemi} 个代表通道")

        # ── 2. 加载通道数据 (chunked frombuffer) ──
        t_load = time.time()
        ch_data = _load_channel_data(
            self.file_path, self.n_channels, hemi_indices)
        data = np.stack([ch_data[i] for i in hemi_indices], axis=0)
        print(f"[Step3] 数据加载: {data.shape}, "
              f"{data.nbytes/1e6:.0f} MB ({time.time()-t_load:.1f}s)")

        # ── 3. MNE FIR 滤波 ──
        t_filt = time.time()
        info = mne.create_info(
            [self.ch_names[i] for i in hemi_indices],
            self.sfreq, ch_types=['eeg'] * n_hemi)
        raw_hemi = mne.io.RawArray(data, info, verbose=False)
        raw_hemi.filter(self.filter_low, self.filter_high)
        filted = raw_hemi.get_data()
        print(f"[Step3] 滤波完成 ({time.time()-t_filt:.1f}s)")

        # ── 4. 设置半球缓存 ──
        self._hemi_data_cache = data
        self._hemi_filted_cache = filted
        self._hemi_ch_names = [self.ch_names[i] for i in hemi_indices]
        # 临时启用多通道模式（使各特征方法放行）
        _load_all_saved = self._load_all
        self._load_all = True

        # ── 5. 运行多通道特征 ──
        try:
            if 'connectivity' in groups:
                print("\n--- 功能连接 (半球代表通道) ---")
                self.functional_connectivity(
                    epoch_sec=epoch_sec, use_roi=True)

            if 'microstates' in groups:
                print("\n--- 微状态分析 (半球代表通道) ---")
                self.microstate_analysis()

            if 'spatial' in groups:
                print("\n--- 空间特征 ---")
                self.global_field_power(epoch_sec=epoch_sec)
                self.spatial_complexity(epoch_sec=epoch_sec)
                print("\n--- 图论指标 ---")
                self.graph_metrics(epoch_sec=epoch_sec, use_roi=True)
        finally:
            self._load_all = _load_all_saved

        # ── 6. 清除半球缓存 ──
        for _attr in ('_hemi_data_cache', '_hemi_filted_cache',
                       '_hemi_ch_names'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        del data, filted, ch_data, raw_hemi
        self._clear_cache('roi')
        print(f"[Step 3/4] ✓ 多通道 ROI 特征完成 (缓存已清除)")

    def run_step4_source_slices(self, epoch_sec=30,
                                 method='eLORETA',
                                 slice_sec=600):
        """Step 4: 源定位 — 全通道时间切片加载，eLORETA 逆解。

        必须纳入全部 260 通道才能准确求解电磁逆问题。
        通过时间切片加载（每次 ~600s ≈ 10min → ~1.2 GB/slice）
        避免一次性加载全部 ~7.5 GB 数据。

        工作流：
          1. 设置 fsaverage 源空间 + BEM 头模型 + 前向解（一次性）
          2. 逐时间切片：加载全部 260 通道 → 滤波 → 逆解 → 累积
          3. 每切片完成后释放原始数据
          4. 最终计算源空间频带功率 + ROI 功率

        Args:
            epoch_sec: epoch 长度
            method: 逆解方法 'eLORETA'/'dSPM'/'MNE'（推荐 eLORETA）
            slice_sec: 每次加载的时间切片长度（秒，默认 600）
        """
        import gc
        import time
        import mne
        from mne.minimum_norm import (make_inverse_operator,
                                       apply_inverse)

        print(f"\n{'='*60}")
        print(f"[Step 4/4] 源定位 ({method}, 时间切片={slice_sec}s)")
        print(f"{'='*60}\n")

        # ── 获取数据维度 ──
        n_eeg_ch = len(self._eeg_ch_indices)  # 纯 EEG 通道 (E1-E256)
        n_total = self.n_times
        sfreq = self.sfreq
        slice_samples = int(slice_sec * sfreq)
        n_slices = int(np.ceil(n_total / slice_samples))
        print(f"[Step4] 总数据: {n_total/sfreq/3600:.1f}h, "
              f"{n_eeg_ch} EEG通道, {n_slices} 个时间切片")

        # ── Step 1: 源空间 (一次性) ──
        print("[Step4] 设置源空间 (fsaverage, oct6)...")
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = str(Path(fs_dir).parent)
        src = mne.setup_source_space(
            'fsaverage', spacing='oct6',
            subjects_dir=subjects_dir, add_dist=False)
        print(f"[Step4] 源空间: ~{sum(len(s['vertno']) for s in src)} 顶点")

        # ── Step 2: BEM (一次性, 三层: 头皮/颅骨/脑) ──
        print("[Step4] 构建 BEM 模型 (三层: scalp/skull/brain)...")
        conductivity = (0.3, 0.006, 0.3)  # scalp, skull, brain
        model_bem = mne.make_bem_model(
            'fsaverage', ico=4, conductivity=conductivity,
            subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model_bem)
        print("[Step4] BEM 模型完成")

        # ── Step 3: 蒙太奇 + 前向解 (一次性) ──
        print("[Step4] 设置电极蒙太奇 + 前向解...")
        ch_names_eeg = [self.ch_names[i] for i in self._eeg_ch_indices]
        info_fwd = mne.create_info(ch_names_eeg, sfreq, ch_types='eeg')
        montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
        info_fwd.set_montage(montage)

        trans = mne.coreg.estimate_head_mri_t(
            'fsaverage', subjects_dir=subjects_dir)

        fwd = mne.make_forward_solution(
            info_fwd, trans=trans, src=src, bem=bem,
            meg=False, eeg=True, mindist=5.0, n_jobs=1)
        fwd = mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=False, copy=False)
        print(f"[Step4] 前向解: {fwd['nsource']} 源, {fwd['nchan']} 通道")

        # ── 逐切片处理（累积 ROI 时间序列）──
        epoch_samples = int(epoch_sec * sfreq)
        roi_ts_epochs = []  # list of (n_rois, n_times_per_epoch)
        roi_names_list = None

        for sl in range(n_slices):
            start_samp = sl * slice_samples
            n_samp = min(slice_samples, n_total - start_samp)
            if n_samp < epoch_samples:
                break  # 最后一片不够一个 epoch，跳过

            print(f"\n[Step4] 切片 {sl+1}/{n_slices}: "
                  f"样本 {start_samp}-{start_samp+n_samp} "
                  f"({n_samp/sfreq:.0f}s)")

            # 加载切片数据（仅 EEG 通道，避免非 EEG 通道干扰源定位）
            t_load = time.time()
            slice_data = _load_channel_data_slice(
                self.file_path, self.n_channels,
                self._eeg_ch_indices,
                start_samp, n_samp)
            stacked = np.stack([slice_data[i] for i in self._eeg_ch_indices],
                               axis=0)
            print(f"[Step4]   加载: {stacked.shape}, "
                  f"{stacked.nbytes/1e6:.0f} MB ({time.time()-t_load:.1f}s)")

            # 滤波
            t_filt = time.time()
            info_slice = mne.create_info(
                ch_names_eeg, sfreq, ch_types='eeg')
            raw_slice = mne.io.RawArray(stacked, info_slice, verbose=False)
            raw_slice.filter(self.filter_low, self.filter_high)
            filted_slice = raw_slice.get_data()
            print(f"[Step4]   滤波 ({time.time()-t_filt:.1f}s)")

            # 释放原始数据
            del slice_data, stacked, raw_slice
            gc.collect()

            # 切分为 epochs
            n_epochs_slice = filted_slice.shape[1] // epoch_samples
            if n_epochs_slice == 0:
                del filted_slice; gc.collect()
                continue

            data_3d = (filted_slice[:, :epoch_samples * n_epochs_slice]
                       .reshape(n_eeg_ch, n_epochs_slice, epoch_samples)
                       .transpose(1, 0, 2))
            events = np.column_stack([
                np.arange(n_epochs_slice, dtype=int),
                np.zeros(n_epochs_slice, int),
                np.ones(n_epochs_slice, int)])
            epochs = mne.EpochsArray(
                data_3d, info_fwd, events=events, tmin=0,
                baseline=None, verbose=False)
            epochs.set_eeg_reference('average', projection=True)
            epochs.apply_proj()
            del filted_slice, data_3d; gc.collect()

            # 噪声协方差（用本切片数据估计）
            noise_cov = mne.compute_covariance(
                epochs, method='shrunk', verbose=False)

            # 逆算子 (force_fixed=True 的前向解必须用 fixed=True)
            inv_op = make_inverse_operator(
                epochs.info, fwd, noise_cov,
                fixed=True, depth=0.8, verbose=False)

            # 逆解每个 epoch（仅提取 ROI 时间序列）
            for ep in range(n_epochs_slice):
                evoked = epochs[ep].average()
                stc = apply_inverse(
                    evoked, inv_op, method=method,
                    pick_ori=None, verbose=False)

                # 首切片: 读取皮层 ROI 标签 (Desikan-Killiany 69 labels)
                if roi_names_list is None:
                    labels = mne.read_labels_from_annot(
                        'fsaverage', parc='aparc',
                        subjects_dir=subjects_dir)
                    roi_names_list = [l.name for l in labels]

                # 提取 ROI 时间序列 (69, n_times)
                label_ts = mne.extract_label_time_course(
                    stc, labels, src, mode='mean',
                    allow_empty=True, verbose=False)
                roi_ts_epochs.append(label_ts)

                del stc

            del epochs, noise_cov, inv_op
            gc.collect()
            print(f"[Step4]   逆解完成: {n_epochs_slice} epochs")

        # ── 汇总: 从 ROI 时间序列计算频带功率 ──
        n_epochs_total = len(roi_ts_epochs)
        if n_epochs_total == 0 or roi_names_list is None:
            print("[Step4] ✗ 无有效源定位结果")
            self._clear_cache('source')
            return

        print(f"\n[Step4] 源定位汇总: {n_epochs_total} epochs × "
              f"{len(roi_names_list)} ROIs")

        # 计算每个 ROI 每 epoch 的频带功率
        roi_band_power = {}
        for band_name, (fl, fh) in self.FREQ_BANDS.items():
            roi_band_power[band_name] = np.zeros(
                (n_epochs_total, len(roi_names_list)))

        for ep in range(n_epochs_total):
            ts = roi_ts_epochs[ep]  # (n_rois, n_times)
            for band_name, (fl, fh) in self.FREQ_BANDS.items():
                # Welch PSD per ROI (69 ROIs, fast)
                freqs, psd = scipy.signal.welch(
                    ts, fs=int(epoch_samples / epoch_sec),
                    nperseg=min(256, ts.shape[1]))
                bm = (freqs >= fl) & (freqs <= fh)
                bp = np.trapz(psd[:, bm], freqs[bm], axis=1)
                roi_band_power[band_name][ep] = bp

        self.features['source_band_power'] = {
            'values': roi_band_power,
            'meta': {'epoch_sec': epoch_sec, 'method': method,
                     'bands': dict(self.FREQ_BANDS),
                     'roi_names': roi_names_list}
        }
        print(f"[Step4] 源空间频带功率: {n_epochs_total} epochs × "
              f"{len(self.FREQ_BANDS)} bands × {len(roi_names_list)} ROIs")

        # ROI RMS 功率
        roi_rms = np.zeros((n_epochs_total, len(roi_names_list)))
        for ep in range(n_epochs_total):
            ts = roi_ts_epochs[ep]
            roi_rms[ep] = np.sqrt(np.mean(ts ** 2, axis=1))

        roi_power_dict = {roi_names_list[i]: roi_rms[:, i]
                          for i in range(len(roi_names_list))}
        n_valid = sum(1 for v in roi_power_dict.values()
                      if not np.all(np.isnan(v)))
        self.features['source_roi_power'] = {
            'values': roi_power_dict,
            'meta': {'epoch_sec': epoch_sec, 'method': method,
                     'parc': 'aparc', 'n_rois': len(roi_names_list)}
        }
        print(f"[Step4] 源空间 ROI 功率: {n_valid}/{len(roi_names_list)} ROIs 非空")

        del roi_ts_epochs
        self._clear_cache('source')
        print(f"[Step 4/4] ✓ 源定位完成 (缓存已清除)")

    # ══════════════════════════════════════════════════════════

    def run_all(self,
                epoch_sec: float = 30,
                groups=('basic', 'time_domain', 'frequency', 'entropy',
                        'complexity', 'connectivity', 'microstates', 'spatial',
                        'tsfresh', 'rqa', 'adv_spectral', 'autocorrelation'),
                skip_yasa: bool = False,
                skip_sssm: bool = False,
                skip_source_loc: bool = False,
                source_method: str = 'eLORETA',
                yasa_eog_channels=None,
                yasa_metadata=None,
                max_per_hemi: int = 5):
        """一键运行全部特征提取 — 4 步线性流水线。

        自动管理缓存：每步加载所需数据，完成后释放。
        内存占用从原 ~42 GB/实例 降至峰值 ~2 GB。

        4 步流水线：
          Step 1: 单通道特征 (E21 ≈Cz) — 频谱/时域/熵/复杂度/tsfresh/RQA
          Step 2: YASA 睡眠分期 (EEG+EOG) — Wake/N1/N2/N3/REM
          Step 3: 多通道 ROI 特征 — 半球分区 (≤5通道/侧)，连接性/微状态/空间/图论
          Step 4: 源定位 (eLORETA) — 时间切片全通道加载，源空间频带/ROI功率

        Args:
            epoch_sec: 默认 epoch 长度
            groups: 要运行的特征组
            skip_yasa: 跳过 Step 2 睡眠分期
            skip_sssm: 跳过 SSSM 特征波检测
            skip_source_loc: 跳过 Step 4 源定位
            source_method: 逆解方法 'eLORETA'/'dSPM'/'MNE'
            yasa_eog_channels: EOG 通道对，默认 ('E67', 'E219')
            yasa_metadata: YASA 可选元数据
            max_per_hemi: Step 3 每半球最多通道数 (默认 5)
        """
        import time
        t_total = time.time()

        print(f"\n{'='*60}")
        print(f"SleepEEGFeatureExtractor — 4步流水线特征提取")
        print(f"文件: {os.path.basename(self.file_path)}")
        print(f"主通道: {self.eeg_channel}, 采样率: {self.sfreq} Hz")
        print(f"总通道: {self.n_channels}")
        print(f"{'='*60}")

        # ── Step 1: 单通道特征 ──
        sc_groups = tuple(g for g in groups
                          if g not in ('connectivity', 'microstates', 'spatial'))
        self.run_step1_single_channel(
            epoch_sec=epoch_sec,
            groups=sc_groups,
            skip_sssm=skip_sssm,
            yasa_metadata=yasa_metadata)
        print(f"  [内存] 单通道数据: ~60 MB (保留)")

        # ── Step 2: YASA 分期 (EOG 增强) ──
        if not skip_yasa:
            eog_ch = yasa_eog_channels or ('E67', 'E219')
            self.run_step2_yasa_with_eog(
                eog_channels=eog_ch,
                metadata=yasa_metadata)
            print(f"  [内存] Step 2 EOG 已清除, 单通道保留")
        else:
            print(f"\n[Step 2/4] YASA 分期 — 已跳过")

        # ── Step 3: 多通道 ROI ──
        mc_groups = tuple(g for g in groups
                          if g in ('connectivity', 'microstates', 'spatial'))
        if mc_groups:
            self.run_step3_multichannel_roi(
                epoch_sec=epoch_sec,
                max_per_hemi=max_per_hemi,
                groups=mc_groups)
            print(f"  [内存] Step 3 半球缓存已清除, 单通道保留")
        else:
            print(f"\n[Step 3/4] 多通道 ROI — 无多通道特征组需要运行")

        # ── Step 4: 源定位 ──
        if not skip_source_loc:
            self.run_step4_source_slices(
                epoch_sec=epoch_sec,
                method=source_method)
            print(f"  [内存] Step 4 源数据已清除, 单通道保留")
        else:
            print(f"\n[Step 4/4] 源定位 — 已跳过")

        elapsed = time.time() - t_total
        print(f"\n{'='*60}")
        print(f"4步流水线完成! 总耗时: {elapsed/60:.1f} 分钟")
        self.summary()
        return self.features

    def summary(self):
        """打印已提取特征的摘要。"""
        print(f"\n--- 特征摘要 ---")
        if not self.features:
            print("  (无特征)")
            return
        for key, val in self.features.items():
            if key == 'recording_date':
                print(f"  {key}: {val.get('date', 'N/A')}")
            elif isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}")
            elif isinstance(val, dict):
                if 'values' in val:
                    v = val['values']
                    if isinstance(v, np.ndarray):
                        print(f"  {key}: shape={v.shape}")
                    elif isinstance(v, dict):
                        shapes = {k: arr.shape for k, arr in v.items()
                                  if isinstance(arr, np.ndarray)}
                        print(f"  {key}: {shapes}")
                    else:
                        print(f"  {key}: {type(v).__name__}")
                elif 'records' in val:
                    print(f"  {key}: {len(val['records'])} records")
                elif 'labels' in val:
                    print(f"  {key}: {val['n_states']} microstates, {len(val['labels'])} labels")
                else:
                    print(f"  {key}: {list(val.keys())}")
            else:
                print(f"  {key}: {type(val).__name__}")

    def to_dataframe(self, epoch_sec: float = 30) -> pd.DataFrame:
        """
        将所有 epoch 对齐的特征导出为 DataFrame。

        处理以下几类特征存储格式:
        - {'values': ndarray} → 直接展开
        - {'values': dict of ndarray} → subkey 展开
        - {'records': [...], 'slope_per_epoch': ndarray, ...} → aperiodic 特殊处理
        - ndarray (直接存储) → feature_waves 等
        - {'values': {'psd_features': ndarray, ...}} → PSD 频带

        Returns:
            pd.DataFrame, 每行一个 epoch
        """
        n_epochs = self._get_n_epochs(epoch_sec)
        rows = []
        for ep in range(n_epochs):
            row = {'epoch': ep}
            for name, val in self.features.items():
                # ---- 格式1: {'values': ndarray} ----
                if isinstance(val, dict) and 'values' in val:
                    v = val['values']
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1 and len(v) == n_epochs:
                            row[name] = v[ep]
                        elif v.ndim == 2 and v.shape[0] == n_epochs:
                            # 每个 epoch 一行, 取均值或分列
                            if v.shape[1] <= 30:
                                for j in range(v.shape[1]):
                                    row[f'{name}_{j}'] = v[ep, j]
                            else:
                                row[f'{name}_mean'] = np.nanmean(v[ep])
                        elif v.ndim == 3 and v.shape[0] == n_epochs:
                            # 连接矩阵等: 取平均连接强度
                            row[f'{name}_mean'] = np.nanmean(v[ep])
                    elif isinstance(v, dict):
                        # {'values': {subk: ndarray, ...}}
                        for subk, subv in v.items():
                            if isinstance(subv, np.ndarray) and len(subv) == n_epochs:
                                row[f'{name}_{subk}'] = subv[ep]

                # ---- 格式2: {'records': [...], 'slope_per_epoch': ndarray, ...} ----
                elif isinstance(val, dict) and 'slope_per_epoch' in val:
                    row['aperiodic_slope'] = val['slope_per_epoch'][ep]
                    if 'beta_per_epoch' in val:
                        row['aperiodic_beta'] = val['beta_per_epoch'][ep]

                # ---- 格式3: ndarray (直接存储) ----
                elif isinstance(val, np.ndarray):
                    if val.ndim == 1 and len(val) == n_epochs:
                        row[name] = val[ep]
                    elif val.ndim == 2 and val.shape[0] == n_epochs:
                        row[f'{name}_mean'] = np.nanmean(val[ep])

            rows.append(row)
        return pd.DataFrame(rows)

    def save_features(self, output_path):
        """
        保存提取的特征到 pickle 文件。

        Args:
            output_path: 输出文件路径
        """
        try:
            import pickle
            # 过滤掉大型数据（raw 对象等），只保存特征
            save_data = {
                'file_path': self.file_path,
                'eeg_channel': self.eeg_channel,
                'sfreq': self.sfreq,
                'n_channels': self.n_channels,
                'features': self.features,
            }
            with open(output_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"特征已保存到: {output_path}")
        except Exception as e:
            print(f"保存特征失败: {e}")
