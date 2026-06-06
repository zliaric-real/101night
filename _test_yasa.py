import mne
import os
import yasa
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import scipy
from sssm_07.sssm.sssm import Model
import unittest

import os

from feature_101night_analy import SleepEEGFeatureExtractor

# 定义两个路径
paths = [
    r'E:\idea\101night',
    r'I:\101Night'
]

# 初始化列表
raw_data_list = []
processed_files = []

# 遍历所有路径
for path in paths:
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        continue
    
    # 只遍历当前目录，不进入子目录
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        
        # 检查是否为目录且以.mff结尾
        if os.path.isdir(full_path) and full_path.lower().endswith('.mff'):
            try:
                processed_files.append(full_path)
                print(f"成功读取: {full_path}")
            except Exception as e:
                print(f"读取文件夹 {full_path} 时出错: {e}")

# 打印统计信息
print(f"\n总共找到 {len(processed_files)} 个.mff文件夹")


extractor = SleepEEGFeatureExtractor(file_path=processed_files[0])

sls = extractor.sleep_stages_yasa()

print(sls.hypno)

# class SleepEEGFeatureExtractor:
#     """
#     睡眠脑电特征提取器
#
#     该类用于从睡眠脑电数据中提取各种特征，包括睡眠分期、时间信息、
#     特征波和非周期特征等。
#
#     Attributes:
#         raw: 原始脑电数据对象
#         sfreq: 采样频率
#         data: EEG数据数组
#         ch_names: 通道名称列表
#     """
#
#     def __init__(self, file_path, eeg_channel='E21', eog_channel='E61'):
#         """
#         初始化特征提取器
#
#         Args:
#             file_path: .mff文件路径
#             eeg_channel: 用于分析的主要EEG通道名称
#             eog_channel: 眼电通道名称 (传给YASA提升Wake/REM区分)
#         """
#         self.file_path = file_path
#
#
#         # 加载原始数据
#         raw = mne.io.read_raw_egi(file_path, preload=False)
#
#         # 提取基本信息
#         self.sfreq = int(raw.info['sfreq'])
#
#         # 检查指定通道是否存在
#         if eeg_channel not in raw.ch_names:
#             print(f"警告: 通道 {eeg_channel} 不存在，使用第一个通道")
#             eeg_channel = raw.ch_names[10]
#         if eog_channel is not None and eog_channel not in raw.ch_names:
#             print(f"警告: 眼电通道 {eog_channel} 不存在，不使用EOG")
#             eog_channel = None
#
#         self.eeg_channel = eeg_channel
#         self.eog_channel = eog_channel
#
#         # 构建 staging raw: EEG + EOG (双通道)，用于 YASA 睡眠分期
#         pick_list = [eeg_channel]
#         if eog_channel:
#             pick_list.append(eog_channel)
#         raw.pick(pick_list)
#         raw.load_data()
#
#         self.raw_stage = raw.copy()  # 保留双通道给 YASA
#
#         # 单通道 raw + data (用于特征提取)
#         self.raw = raw.copy().pick([eeg_channel])
#         self.data = self.raw.get_data(units="uV")
#
#         self.filted = self.raw.copy()
#         self.filted = self.filted.filter(0.1, 40)
#
#         print(f"[init] EEG={eeg_channel}, EOG={eog_channel}, "
#               f"channels={len(pick_list)}, sfreq={self.sfreq}Hz, "
#               f"duration={self.data.shape[-1]/self.sfreq/3600:.1f}h")
#
#         # 存储提取的特征
#         self.features = { }
#
#     def sleep_stages_yasa(self, win_sec=30, hypno_freq=1/30):
#         """
#         使用YASA进行自动睡眠分期
#
#         Args:
#             win_sec: 分析窗口长度（秒），默认30秒
#             hypno_freq: 分期频率（Hz），默认为1/30（30秒一个分期）
#
#         Returns:
#             睡眠分期数组（0: Wake, 1: N1, 2: N2, 3: N3, 4: REM）
#         """
#         try:
#             # 使用 raw_stage (EEG+EOG) 进行分期
#             kwargs = dict(eeg_name=self.eeg_channel)
#             if self.eog_channel:
#                 kwargs['eog_name'] = self.eog_channel
#                 print(f"[YASA] 使用 EEG={self.eeg_channel} + EOG={self.eog_channel}")
#             else:
#                 print(f"[YASA] 使用 EEG={self.eeg_channel} (无EOG)")
#
#             sls = yasa.SleepStaging(self.raw_stage, **kwargs)
#
#             # 获取预测的睡眠分期
#             hypno_pred = sls.predict()
#             self.features['sleep_stages'] = {
#                 'stages': hypno_pred,
#                 'stage_labels': {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'},
#                 'epoch_length': win_sec,
#                 'total_epochs': len(hypno_pred)
#             }
#
#             print("睡眠分期计算完成")
#             return hypno_pred
#
#         except Exception as e:
#             print(f"睡眠分期提取失败: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
#
#     def recording_date(self):
#         """
#         提取脑电记录的日期和时间信息
#
#         Returns:
#             包含日期时间信息的字典
#         """
#         try:
#             # 从raw.info中提取记录开始时间
#             meas_date = self.raw.info['meas_date']
#
#             if isinstance(meas_date, tuple):
#                 # 如果meas_date是元组格式
#                 dt_obj = datetime.fromtimestamp(meas_date[0])
#             else:
#                 # 如果meas_date是datetime对象
#                 dt_obj = meas_date
#
#             # 格式化日期时间信息
#             date_info = {
#                 'datetime': dt_obj,
#                 'date': dt_obj.date(),
#                 'time': dt_obj.time(),
#                 'year': dt_obj.year,
#                 'month': dt_obj.month,
#                 'day': dt_obj.day,
#                 'hour': dt_obj.hour,
#                 'minute': dt_obj.minute,
#                 'second': dt_obj.second
#             }
#
#             self.features['recording_date'] = date_info
#             return date_info
#
#         except Exception as e:
#             print(f"日期时间提取失败: {e}")
#             return None
#
#     def psd(self, win_sec=6, **kwargs):
#         """
#         使用 mne_features.compute_pow_freq_bands 提取频带功率特征
#
#         Args:
#             win_sec: 滑动窗口长度（秒），默认 6 秒
#             **kwargs: 传递给 compute_pow_freq_bands 的参数
#
#         Returns:
#             feature_psd dict
#         """
#
#         try:
#             sfreq = self.sfreq
#             data = self.data  # shape: (1, n_samples) or (n_samples,)
#
#             # 确保是 (n_channels, n_times)
#             if data.ndim == 1:
#                 data = data[np.newaxis, :]
#
#             n_channels, n_times = data.shape
#             win_samples = int(win_sec * sfreq)
#
#             if n_times < win_samples:
#                 print("[psd] 数据长度小于一个窗口")
#                 return None
#
#             n_windows = n_times // win_samples
#
#             # EEG 常用频段
#             freq_bands = np.array([0.5, 4, 8, 13, 30, 45])
#
#             psd_features = []
#
#             for w in range(n_windows):
#                 start = w * win_samples
#                 end = start + win_samples
#                 segment = data[:, start:end]
#
#                 feat = compute_pow_freq_bands(
#                     sfreq=sfreq,
#                     data=segment,
#                     freq_bands=freq_bands,
#                     normalize=True,   # 每个频带 / 总功率
#                     log=False,
#                     psd_method='welch',
#                     psd_params=None
#                 )
#
#                 psd_features.append(feat)
#
#             psd_features = np.array(psd_features)  # shape: (n_windows, n_channels * n_bands)
#
#             print("功率谱计算完成")
#
#             # 整理结果
#             feature_psd = {
#                 'win_sec': win_sec,
#                 'n_windows': n_windows,
#                 'freq_bands': freq_bands,
#                 'band_names': ['delta', 'theta', 'alpha', 'beta', 'gamma'],
#                 'psd_features': psd_features,  # 核心输出
#                 'shape': psd_features.shape
#             }
#
#             self.features['feature_psd'] = feature_psd
#             return feature_psd
#
#         except Exception as e:
#             print(f"能量谱提取失败: {e}")
#             return None
#
#
#     def _irasa(self, data_1ch_uv: np.ndarray):
#         """
#         data_1ch_uv: shape (1, n_samples) 或 (n_samples,)，单位 uV
#         """
#         try:
#             # --- 新增：数据清洗步骤 ---
#             # 1. 确保是 1D 数组
#             if data_1ch_uv.ndim == 2 and data_1ch_uv.shape[0] == 1:
#                 data_1ch_uv = data_1ch_uv.ravel()
#             elif data_1ch_uv.ndim == 2:
#                 # 如果是多通道传入，这里只取第一个通道，或者报错
#                 # 根据你的逻辑，这里假设是单通道
#                 data_1ch_uv = data_1ch_uv[0]
#
#             # 2. 去除直流偏移（非常重要，IRASA 对 DC 敏感）
#             data_1ch_uv = data_1ch_uv - np.mean(data_1ch_uv)
#
#             # 3. 检查并替换 inf/nan
#             if np.any(~np.isfinite(data_1ch_uv)):
#                 print("[WARN] 数据包含 inf 或 nan，尝试修复...")
#                 # 用邻近值或中位数填充，或者直接丢弃该段（这里用中位数简单填充）
#                 # 更稳妥的做法是标记该 epoch 为无效
#                 data_1ch_uv = np.nan_to_num(data_1ch_uv, nan=np.nanmedian(data_1ch_uv), posinf=np.nanmedian(data_1ch_uv), neginf=np.nanmedian(data_1ch_uv))
#                 # 如果修复后还有问题，直接返回 None
#                 if np.any(~np.isfinite(data_1ch_uv)):
#                     return None, None, None
#
#             # 4. 检查是否全零或方差极小（会导致除以零或 log(0) 错误）
#             if np.var(data_1ch_uv) < 1e-10:
#                 print("[WARN] 数据方差为零或极小，跳过 IRASA")
#                 return None, None, None
#
#             # --- 结束新增 ---
#
#             out = yasa.irasa(
#                 data=data_1ch_uv, # 确保传入的是干净的 1D 数组
#                 sf=self.sfreq,
#                 ch_names=[self.eeg_channel], # 如果 data 是 1D，ch_names 可能需要调整或省略，视 yasa 版本而定
#                 band=(0.3, 35),
#                 hset=[1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45,
#                       1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85],
#                 return_fit=True,
#                 win_sec=4,
#                 verbose=False,
#             )
#
#             freqs = out[0]
#             psd_aper = out[1]
#             psd_osc = out[2]
#             return freqs, psd_aper, psd_osc
#
#         except Exception as e:
#             print(f"[WARN] IRASA failed: {e}")
#             return None, None, None
#
#     def feature_waves(self, **kwargs):
#         try:
#             raw_sssm = self.filted.copy()
#             raw_sssm = raw_sssm.resample(100)
#             data_sssm = raw_sssm.get_data(units="uV")
#             model = Model(device='cpu')
#             model.predict(data_sssm,step=300)
#             pred_labels = model.pred
#             window_indices = np.arange(pred_labels.shape[1])  # 窗口索引
#             self.features['feature_waves'] = pred_labels
#             print(f"特征波提取完成")
#             return pred_labels
#
#         except Exception as e:
#             print(f"特征波提取失败: {e}")
#             return None
#
#
#     def aperiodic(self,
#               epoch_sec: int = 30,
#               fit_band: tuple = (0.3, 35.0),
#               min_valid_bins: int = 3):
#         """
#         把 EEG 切成 epoch_sec 的片段 -> IRASA -> log-log fit -> slope(beta)
#         结果存入 self.features['aperiodic_features']
#         """
#
#         data = np.asarray(self.data)
#         sf = int(self.sfreq)
#
#         if data.ndim == 2 and data.shape[0] == 1:
#             data = data[0]  # (n_samples,)
#         elif data.ndim != 1:
#             raise ValueError("self.data 预期为 (1, n) 或 (n,) for single-channel.")
#
#         epoch_samples = int(epoch_sec * sf)
#         n_total = data.size
#         n_epochs = n_total // epoch_samples
#
#         if n_epochs < 1:
#             print("[aperiodic] 数据太短，连一个 30s epoch 都没有")
#             self.features['aperiodic_features'] = None
#             return None
#
#         rec = []  # 每条 epoch 的结果
#
#         for k in range(n_epochs):
#             seg = data[k * epoch_samples:(k + 1) * epoch_samples]
#
#             if np.any(~np.isfinite(seg)):
#                 print(f"[aperiodic] Epoch {k} 包含无效值，跳过")
#                 rec.append({"epoch": k, "slope_loglog": np.nan, "beta": np.nan, "intercept_loglog": np.nan, "ok": False})
#                 continue
#
#             freqs, psd_ap, _ = self._irasa(seg)
#
#             # 统一失败/缺结果的标记
#             if freqs is None or psd_ap is None:
#                 rec.append({
#                     "epoch": k,
#                     "slope_loglog": np.nan,
#                     "beta": np.nan,
#                     "intercept_loglog": np.nan,
#                     "ok": False,
#                 })
#                 continue
#
#             # ---- log-log fit ----
#             # psd_aperiodic 可能是 (n_freqs,) 或 (1, n_freqs)
#             psd_ap = np.asarray(psd_ap).ravel()
#
#             f = np.asarray(freqs)
#             p = psd_ap
#
#             # 基本有效性
#             mask = (
#                 (f > 0.0) &
#                 np.isfinite(f) &
#                 np.isfinite(p) &
#                 (p > 0.0)
#             )
#
#             # 频带裁剪
#             lo, hi = fit_band
#             mask &= (f >= lo) & (f <= hi)
#
#             if mask.sum() < min_valid_bins:
#                 # 频点太少，不拟合
#                 rec.append({
#                     "epoch": k,
#                     "slope_loglog": np.nan,
#                     "beta": np.nan,
#                     "intercept_loglog": np.nan,
#                     "ok": False,
#                 })
#                 continue
#
#             X = np.log10(f[mask])
#             Y = np.log10(p[mask])
#
#             # OLS: Y = a + b*X
#             A = np.column_stack([np.ones_like(X), X])
#             (b, a), *_ = np.linalg.lstsq(A, Y, rcond=None)
#
#             info = dict(
#                 epoch=k,
#                 slope_loglog=b,      # log-log 直线斜率 b
#                 beta=-b,              # 1/f^beta 的指数
#                 intercept_loglog=a,   # log10 空间的截距
#                 fit_lo=lo,
#                 fit_hi=hi,
#                 ok=True,
#             )
#             rec.append(info)
#
#         print("非周期特征计算完成")
#
#         # ---- 汇总保存 ----
#         out = {
#             "epoch_sec": epoch_sec,
#             "fit_band": fit_band,
#             "records": rec,
#             # 方便下游：直接拿到一个 epoch-wise 数组
#             "slope_per_epoch": np.array([r["slope_loglog"] for r in rec]),
#             "beta_per_epoch": np.array([r["beta"] for r in rec]),
#         }
#
#         self.features['aperiodic_features'] = out
#         return out
#
#     def spectral_entropy(self, win_sec=30,fmin=0.3, fmax=35.0, psd_method='welch'):
#         """
#         计算 30 秒窗口的频谱熵 (Spectral Entropy)
#
#         该方法将数据按 30 秒分段（Epoch），计算每个 Epoch 的频谱熵。
#         结果可以直接与 YASA 的睡眠分期结果一一对应。
#
#         Args:
#             fmin: 计算熵的最低频率 (Hz)，默认 0.5 Hz
#             fmax: 计算熵的最高频率 (Hz)，默认 30 Hz
#             psd_method: PSD 估计方法 ('welch', 'multitaper', 'fft')
#
#         Returns:
#             se_epochs: ndarray, shape (n_epochs, n_channels)
#                       每个 epoch 和通道的频谱熵值
#         """
#         sfreq = self.sfreq
#         data = self.data  # 形状应为 (n_channels, n_times)
#
#         # 确保数据是 2D 的
#         if data.ndim == 1:
#             data = data[np.newaxis, :]
#
#         epoch_len = int(win_sec * sfreq)  # 30秒对应的样本数
#         n_channels, n_times = data.shape
#         n_epochs = n_times // epoch_len
#
#         if n_epochs == 0:
#             print("[Spectral Entropy] 数据长度不足 30 秒")
#             return None
#
#         # 预分配结果数组
#         # compute_spect_entropy 返回的是 (n_channels,)，所以我们按 epoch 循环
#         se_epochs = np.zeros((n_epochs, n_channels))
#
#         print(f"[Spectral Entropy] 开始计算 {n_epochs} 个 30 秒 Epoch 的频谱熵...")
#
#         for ep in range(n_epochs):
#             start = ep * epoch_len
#             end = start + epoch_len
#
#             # 截取当前 epoch 的数据 (n_channels, epoch_len)
#             epoch_data = data[:, start:end]
#
#             try:
#                 # 调用 mne_features 的函数
#                 # 注意：compute_spect_entropy 内部会做 PSD，我们只关心 fmin-fmax 范围内的熵
#                 # 但是该函数目前没有直接的频带限制参数，通常是对全频带计算
#                 # 如果需要限制频带，通常需要先滤波或后处理，但标准用法是直接算
#                 se = compute_spect_entropy(
#                     sfreq=sfreq,
#                     data=epoch_data,
#                     psd_method=psd_method
#                 )
#                 se_epochs[ep, :] = se
#
#             except Exception as e:
#                 print(f"  Epoch {ep} 计算失败: {e}")
#                 se_epochs[ep, :] = np.nan
#
#         print("计算完成。")
#
#         # 存入特征字典
#         if 'spectral_entropy' not in self.features:
#             self.features['spectral_entropy'] = {}
#
#         self.features['spectral_entropy'] = {
#             'epoch_sec': win_sec,
#             'values': se_epochs,  # shape: (n_epochs, n_channels)
#             'n_epochs': n_epochs,
#             'n_channels': n_channels
#         }
#
#         return se_epochs
#
#     def hj_mobility(self, l_freq=0.3, h_freq=35.0,epoch_dur=30):
#         sfreq = self.sfreq
#         data = self.filted.get_data(units="uV")
#
#         if data.ndim == 1:
#             data = data[np.newaxis, :]
#
#         n_channels, n_times = data.shape
#         epoch_len = int(epoch_dur * sfreq)
#         n_epochs = n_times // epoch_len
#
#         if n_epochs == 0:
#             print("[Hjorth Mobility] 数据长度不足 30 秒")
#             return None
#         mobility_epochs = np.zeros((n_epochs, n_channels))
#
#         print(f"[Hjorth Mobility] 开始计算 {n_epochs} 个 30 秒 Epoch...")
#
#         for ep in range(n_epochs):
#             start = ep * epoch_len
#             end = start + epoch_len
#
#             # 截取滤波后的数据
#             epoch_data = data[:, start:end]
#
#             try:
#                 # mne_features 的函数
#                 mob = compute_hjorth_mobility(epoch_data)
#                 mobility_epochs[ep, :] = mob
#             except Exception as e:
#                 print(f"  Epoch {ep} 计算失败: {e}")
#                 mobility_epochs[ep, :] = np.nan
#
#         print("计算完成。")
#
#         # ---------------------------------------------------------
#         # 4. 存储结果
#         # ---------------------------------------------------------
#         if 'hjorth_mobility' not in self.features:
#             self.features['hjorth_mobility'] = {}
#
#         self.features['hjorth_mobility'] = {
#             'epoch_sec': 30,
#             'values': mobility_epochs,  # shape: (n_epochs, n_channels)
#             'filter': f'{l_freq}-{h_freq} Hz FIR',
#             'n_epochs': n_epochs
#         }
#
#         return mobility_epochs
#
#
#
#     def save_features(self, output_path):
#         """
#         保存提取的特征到文件
#
#         Args:
#             output_path: 输出文件路径
#         """
#         try:
#             import pickle
#             with open(output_path, 'wb') as f:
#                 pickle.dump({
#                     'file_path': self.file_path,
#                     'eeg_channel': self.eeg_channel,
#                     'sfreq': self.sfreq,
#                     'features': self.features
#                 }, f)
#             print(f"特征已保存到: {output_path}")
#         except Exception as e:
#             print(f"保存特征失败: {e}")
#





