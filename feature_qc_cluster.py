# -*- coding: utf-8 -*-
"""
feature_qc_cluster.py — 脑电特征质量控制 + 归一化 + 无监督聚类 (k=5)
==================================================================
对全部 101 夜的 epoch 级脑电特征进行系统化质量控制、归一化预处理，
并使用 k-means (k=5) 进行无监督聚类，发现睡眠脑电的自然结构。

流程:
  1. 加载: 从 features_night*.pkl 或 features_night*.csv 加载所有夜晚
  2. QC:
     a. 移除 NaN 占比 >50% 的特征列
     b. 移除 NaN 占比 >30% 的 epoch (样本)
     c. 中位数填充剩余 NaN
     d. 移除零方差 / 近零方差特征
     e. 移除高度离群 epoch (IQR × 3.0)
  3. 归一化: RobustScaler (对离群值鲁棒)
  4. 聚类: KMeans k=5, n_init=20
  5. 评估: Silhouette, Davies-Bouldin, Calinski-Harabasz
  6. 可视化:
     - t-SNE 投影 + 聚类着色
     - 聚类大小饼图
     - 每类 Top-20 判别特征热力图
     - 聚类在夜晚间的分布 (每夜的聚类比例)
  7. 输出:
     - clustered_features.csv: 原始特征 + cluster_label
     - cluster_centers.csv: 聚类中心 (归一化空间)
     - cluster_report.txt: 详细文本报告

依赖:
  conda activate eeg_101night
  pip install scikit-learn  (如未安装)

用法:
  python feature_qc_cluster.py
  python feature_qc_cluster.py --data-dir ./features  # 指定特征目录
"""

import os
import re
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# ── 路径 ──
OUTPUT_DIR = Path(__file__).resolve().parent
PLOT_DIR = OUTPUT_DIR / "sleep_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── QC 参数 ──
MAX_FEATURE_NAN_RATIO = 0.50   # 特征 NaN 比例超过此值 → 删除该特征
MAX_SAMPLE_NAN_RATIO = 0.30    # 样本 NaN 比例超过此值 → 删除该 epoch
IQR_MULTIPLIER = 3.0           # IQR 离群值倍数
VAR_THRESHOLD = 1e-8           # 近零方差阈值

# ── 聚类参数 ──
N_CLUSTERS = 5
RANDOM_STATE = 42

# ── 非特征列 (不参与聚类) ──
META_COLUMNS = {'night', 'epoch', 'date', 'Unnamed: 0'}


# ═══════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════

def find_feature_files(data_dir=None):
    """在指定目录中查找所有特征文件 (.pkl 或 .csv)。"""
    if data_dir is None:
        data_dir = OUTPUT_DIR

    data_dir = Path(data_dir)
    pkl_files = sorted(data_dir.glob("features_night*.pkl"))
    csv_files = sorted(data_dir.glob("features_night*.csv"))

    # 去重: 同夜的两格式只保留 .pkl
    pkl_nights = set()
    for f in pkl_files:
        m = re.search(r'night(\d+)', f.stem)
        if m:
            pkl_nights.add(int(m.group(1)))

    csv_files_filtered = []
    for f in csv_files:
        m = re.search(r'night(\d+)', f.stem)
        if m and int(m.group(1)) not in pkl_nights:
            csv_files_filtered.append(f)

    return pkl_files + csv_files_filtered


def load_features_from_pkl(pkl_path):
    """从 .pkl 加载特征并转为 DataFrame。

    Args:
        pkl_path: Path to .pkl file

    Returns:
        (pd.DataFrame, meta_dict) or (None, None)
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  WARNING: 无法加载 {pkl_path.name}: {e}")
        return None, None

    night = _extract_night_from_path(pkl_path)
    features = data.get("features", {})
    if not features:
        print(f"  WARNING: {pkl_path.name} 中无特征数据")
        return None, None

    # 确定 epoch 数: 从第一个特征推断
    n_epochs = None
    for name, val in features.items():
        if isinstance(val, dict) and 'values' in val:
            v = val['values']
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                n_epochs = v.shape[0] if v.ndim <= 2 else v.shape[0]
                break

    if n_epochs is None:
        print(f"  WARNING: {pkl_path.name} 无法确定 epoch 数")
        return None, None

    # 展开所有特征为 DataFrame
    rows = []
    for ep in range(n_epochs):
        row = {"night": night, "epoch": ep}
        for name, val in features.items():
            if not isinstance(val, dict) or 'values' not in val:
                continue
            v = val['values']

            if isinstance(v, np.ndarray):
                if v.ndim == 1 and len(v) == n_epochs:
                    row[name] = float(v[ep])
                elif v.ndim == 2 and v.shape[0] == n_epochs:
                    if v.shape[1] <= 50:
                        for j in range(v.shape[1]):
                            row[f"{name}_{j}"] = float(v[ep, j])
                    else:
                        row[f"{name}_mean"] = float(np.nanmean(v[ep]))
                elif v.ndim == 3 and v.shape[0] == n_epochs:
                    row[f"{name}_mean"] = float(np.nanmean(v[ep]))
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if isinstance(subv, np.ndarray) and len(subv) == n_epochs:
                        row[f"{name}_{subk}"] = float(subv[ep])

        # aperiodic 特殊处理
        if 'aperiodic' in features:
            ap = features['aperiodic']
            if isinstance(ap, dict) and 'slope_per_epoch' in ap:
                row['aperiodic_slope'] = float(ap['slope_per_epoch'][ep])

        rows.append(row)

    df = pd.DataFrame(rows)
    meta = {
        "night": night,
        "n_epochs": n_epochs,
        "n_features": len(df.columns) - 2,  # minus night + epoch
    }
    return df, meta


def load_features_from_csv(csv_path):
    """从 .csv 加载特征 DataFrame。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  WARNING: 无法加载 {csv_path.name}: {e}")
        return None, None

    night = _extract_night_from_path(csv_path)
    if "night" not in df.columns:
        df.insert(0, "night", night)

    # 清理 Unnamed 列
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    df = df.drop(columns=unnamed_cols, errors='ignore')

    meta = {
        "night": night,
        "n_epochs": len(df),
        "n_features": len(df.columns) - len(
            [c for c in df.columns if c in META_COLUMNS]),
    }
    return df, meta


def _extract_night_from_path(file_path):
    """从文件名提取夜晚编号。"""
    m = re.search(r'night(\d+)', Path(file_path).stem)
    return int(m.group(1)) if m else None


def load_all_features(data_dir=None):
    """加载所有夜晚的特征数据。

    Returns:
        df_all: pd.DataFrame, 所有夜晚合并
        meta_list: list of dict, 每夜的元数据
    """
    files = find_feature_files(data_dir)
    if not files:
        raise FileNotFoundError(
            "未找到任何特征文件 (features_night*.pkl / *.csv)。\n"
            "请先运行 SleepEEGFeatureExtractor.run_all() + save_features()。"
        )

    print(f"找到 {len(files)} 个特征文件")
    dfs = []
    meta_list = []

    for fp in files:
        night = _extract_night_from_path(fp)
        ftype = fp.suffix

        if ftype == ".pkl":
            df, meta = load_features_from_pkl(fp)
        elif ftype == ".csv":
            df, meta = load_features_from_csv(fp)
        else:
            continue

        if df is not None:
            dfs.append(df)
            meta_list.append(meta)
            nf = meta.get("n_features", len(df.columns))
            print(f"  N{night}: {meta['n_epochs']} epochs, ~{nf} features [{ftype}]")
        else:
            print(f"  N{night}: SKIP (加载失败)")

    if not dfs:
        raise ValueError("所有特征文件加载失败。")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n总计: {len(df_all)} epochs, {len(df_all.columns)} 列 (含 meta 列)")
    return df_all, meta_list


# ═══════════════════════════════════════════════════════════
#  质量控制
# ═══════════════════════════════════════════════════════════

def qc_report(df_all, feature_cols):
    """生成 QC 前的基础统计报告。"""
    n_total_features = len(feature_cols)
    n_total_samples = len(df_all)

    nan_counts = df_all[feature_cols].isna().sum()
    nan_ratios = nan_counts / n_total_samples

    print("\n" + "=" * 60)
    print("质量控制报告 — 处理前")
    print("=" * 60)
    print(f"  总样本 (epochs): {n_total_samples}")
    print(f"  总特征: {n_total_features}")
    print(f"  总 NaN 值: {df_all[feature_cols].isna().sum().sum()}")
    print(f"  NaN 占比: {df_all[feature_cols].isna().sum().sum() / (n_total_features * n_total_samples) * 100:.2f}%")
    print(f"\n  NaN 比例分布 (特征):")
    print(f"    无 NaN:       {(nan_ratios == 0).sum()} 个特征")
    print(f"    0-10% NaN:    {((nan_ratios > 0) & (nan_ratios <= 0.1)).sum()}")
    print(f"    10-30% NaN:   {((nan_ratios > 0.1) & (nan_ratios <= 0.3)).sum()}")
    print(f"    30-50% NaN:   {((nan_ratios > 0.3) & (nan_ratios <= 0.5)).sum()}")
    print(f"    >50% NaN:     {(nan_ratios > 0.5).sum()}  ← 将被移除")
    print(f"    100% NaN:     {(nan_ratios == 1.0).sum()}  ← 将被移除")

    sample_nan_ratios = df_all[feature_cols].isna().mean(axis=1)
    print(f"\n  NaN 比例分布 (epochs):")
    print(f"    >30% NaN:     {(sample_nan_ratios > MAX_SAMPLE_NAN_RATIO).sum()}  ← 将被移除")


def apply_qc(df_all):
    """执行完整的质量控制流水线。

    步骤:
      1. 分离 meta 列和特征列
      2. 移除高 NaN 特征
      3. 移除高 NaN 样本
      4. 中位数填充剩余 NaN
      5. 移除零方差特征
      6. 移除离群样本

    Returns:
        df_clean: 清洗后的 DataFrame (含 meta 列)
        feature_cols: 清洗后的特征列名
        qc_summary: QC 统计摘要
    """
    # 分离 meta 和特征列
    meta_cols = [c for c in df_all.columns if c in META_COLUMNS]
    feature_cols = [c for c in df_all.columns if c not in META_COLUMNS]
    feature_cols = [c for c in feature_cols if df_all[c].dtype in ('float64', 'float32', 'int64', 'int32')]

    print(f"  分离: {len(meta_cols)} meta 列, {len(feature_cols)} 特征列")

    n_samples_start = len(df_all)
    n_features_start = len(feature_cols)
    total_nan_start = df_all[feature_cols].isna().sum().sum()

    # ── Step 1: 移除高 NaN 特征 ──
    nan_ratios = df_all[feature_cols].isna().mean()
    bad_features = nan_ratios[nan_ratios > MAX_FEATURE_NAN_RATIO].index.tolist()
    feature_cols = [c for c in feature_cols if c not in bad_features]
    n_features_removed_1 = len(bad_features)
    print(f"  QC-1: 移除 {n_features_removed_1} 个高 NaN 特征 (>{MAX_FEATURE_NAN_RATIO*100:.0f}%)")

    # ── Step 2: 移除高 NaN 样本 ──
    sample_nan = df_all[feature_cols].isna().mean(axis=1)
    bad_samples = sample_nan > MAX_SAMPLE_NAN_RATIO
    df_clean = df_all[~bad_samples].copy()
    n_samples_removed = bad_samples.sum()
    print(f"  QC-2: 移除 {n_samples_removed} 个高 NaN epoch (>{MAX_SAMPLE_NAN_RATIO*100:.0f}%)")

    # ── Step 3: 中位数填充剩余 NaN ──
    remaining_nan = df_clean[feature_cols].isna().sum().sum()
    if remaining_nan > 0:
        # 每列用该列的中位数填充
        medians = df_clean[feature_cols].median()
        df_clean[feature_cols] = df_clean[feature_cols].fillna(medians)
        print(f"  QC-3: 中位数填充 {remaining_nan} 个剩余 NaN")

    # ── Step 4: 移除零方差 / 近零方差特征 ──
    variances = df_clean[feature_cols].var()
    zero_var = variances[variances < VAR_THRESHOLD].index.tolist()
    feature_cols = [c for c in feature_cols if c not in zero_var]
    print(f"  QC-4: 移除 {len(zero_var)} 个近零方差特征")

    # ── Step 5: 移除高度离群样本 (IQR 法, 基于主要特征的离群分数) ──
    # 为每个样本计算离群分数 (有多少个特征超出 IQR × 3.0)
    Q1 = df_clean[feature_cols].quantile(0.25)
    Q3 = df_clean[feature_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - IQR_MULTIPLIER * IQR
    upper = Q3 + IQR_MULTIPLIER * IQR

    outlier_count = ((df_clean[feature_cols] < lower) |
                     (df_clean[feature_cols] > upper)).sum(axis=1)
    # 如果一个样本超过 30% 的特征是离群值，标记为离群样本
    outlier_threshold = int(len(feature_cols) * 0.30)
    extreme_outliers = outlier_count > outlier_threshold
    df_clean = df_clean[~extreme_outliers].copy()
    print(f"  QC-5: 移除 {extreme_outliers.sum()} 个极端离群 epoch "
          f"(>30% 特征超出 IQR×{IQR_MULTIPLIER})")

    # ── 最终统计 ──
    n_samples_end = len(df_clean)
    n_features_end = len(feature_cols)
    total_nan_end = df_clean[feature_cols].isna().sum().sum()

    qc_summary = {
        "samples_before": n_samples_start,
        "samples_after": n_samples_end,
        "samples_removed": n_samples_start - n_samples_end,
        "features_before": n_features_start,
        "features_after": n_features_end,
        "features_removed_nan": n_features_removed_1,
        "features_removed_zero_var": len(zero_var),
        "nan_count_before": int(total_nan_start),
        "nan_count_after": int(total_nan_end),
    }

    print(f"\n  QC 完成:")
    print(f"    样本: {n_samples_start} → {n_samples_end} "
          f"({n_samples_start - n_samples_end} 移除)")
    print(f"    特征: {n_features_start} → {n_features_end} "
          f"({n_features_start - n_features_end} 移除)")
    print(f"    NaN:  {total_nan_start} → {total_nan_end}")

    return df_clean, feature_cols, qc_summary


# ═══════════════════════════════════════════════════════════
#  归一化
# ═══════════════════════════════════════════════════════════

def normalize_features(df, feature_cols):
    """使用 RobustScaler 归一化特征矩阵。

    RobustScaler 使用中位数和 IQR，对离群值鲁棒，适合 EEG 特征。
    """
    from sklearn.preprocessing import RobustScaler

    X = df[feature_cols].values.astype(np.float64)
    scaler = RobustScaler(unit_variance=True)
    X_scaled = scaler.fit_transform(X)

    print(f"  归一化完成: RobustScaler, shape={X_scaled.shape}")
    return X_scaled, scaler, feature_cols


# ═══════════════════════════════════════════════════════════
#  聚类
# ═══════════════════════════════════════════════════════════

def run_clustering(X_scaled, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
    """执行 k-means 聚类并评估。

    Returns:
        labels: 聚类标签 (0 ~ k-1)
        model: 训练好的 KMeans 模型
        metrics: 评估指标 dict
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                  calinski_harabasz_score)

    print(f"\n  执行 KMeans (k={n_clusters}, n_init=20)...")
    model = KMeans(n_clusters=n_clusters, random_state=random_state,
                   n_init=20, max_iter=500)
    labels = model.fit_predict(X_scaled)

    # 评估指标
    metrics = {
        "silhouette": silhouette_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
        "inertia": model.inertia_,
    }

    # 聚类大小
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))

    print(f"\n  聚类完成:")
    print(f"    Silhouette:       {metrics['silhouette']:.4f}")
    print(f"    Davies-Bouldin:   {metrics['davies_bouldin']:.4f} (越低越好)")
    print(f"    Calinski-Harabasz: {metrics['calinski_harabasz']:.1f} (越高越好)")
    print(f"    聚类大小: ", end="")
    for k in sorted(cluster_sizes.keys()):
        pct = cluster_sizes[k] / len(labels) * 100
        print(f"C{k}={cluster_sizes[k]} ({pct:.1f}%)", end="  ")
    print()

    return labels, model, metrics


def characterize_clusters(X_scaled, labels, feature_cols, model, top_n=20):
    """分析每个聚类的判别特征。

    Returns:
        cluster_profiles: dict, cluster_id → {
            'size', 'pct', 'top_features': [(name, z_score), ...]
        }
    """
    centers = model.cluster_centers_
    n_clusters = centers.shape[0]

    # 全局均值和标准差 (用于计算 z-score)
    global_mean = X_scaled.mean(axis=0)
    global_std = X_scaled.std(axis=0)
    global_std[global_std == 0] = 1e-8

    profiles = {}
    for k in range(n_clusters):
        mask = labels == k
        z_scores = (centers[k] - global_mean) / global_std
        # 按绝对值排序，取 top_n
        top_idx = np.argsort(np.abs(z_scores))[::-1][:top_n]
        top_features = [(feature_cols[i], z_scores[i]) for i in top_idx]

        profiles[k] = {
            "size": int(mask.sum()),
            "pct": mask.sum() / len(labels) * 100,
            "center": centers[k],
            "top_features": top_features,
        }

    return profiles


# ═══════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════

def plot_tsne(X_scaled, labels, max_samples=5000):
    """t-SNE 降维到 2D 并着色聚类。"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [SKIP] t-SNE: scikit-learn 版本不支持")
        return

    # 如果样本太多，随机采样
    n = X_scaled.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n, max_samples, replace=False)
        X_sub = X_scaled[idx]
        labels_sub = labels[idx]
        title_suffix = f" (抽样 {max_samples}/{n})"
    else:
        X_sub = X_scaled
        labels_sub = labels
        title_suffix = ""

    print(f"  计算 t-SNE... ({X_sub.shape[0]} 样本)")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                perplexity=30, n_iter=1000, verbose=0)
    X_2d = tsne.fit_transform(X_sub)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.Set2

    n_clusters = len(np.unique(labels))
    colors = [cmap(i % 8) for i in range(n_clusters)]

    # 左: 按聚类着色
    for k in range(n_clusters):
        mask = labels_sub == k
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[k]],
                    s=3, alpha=0.5, label=f'C{k}')
    ax1.set_title(f't-SNE 投影 — 聚类着色{title_suffix}', fontsize=12)
    ax1.legend(markerscale=6, fontsize=8, loc='upper right')
    ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')

    # 右: 密度等高线
    from scipy.stats import gaussian_kde
    for k in range(n_clusters):
        mask = labels_sub == k
        if mask.sum() < 10:
            continue
        try:
            kde = gaussian_kde(X_2d[mask].T)
            x_grid = np.linspace(X_2d[:, 0].min(), X_2d[:, 0].max(), 80)
            y_grid = np.linspace(X_2d[:, 1].min(), X_2d[:, 1].max(), 80)
            Xg, Yg = np.meshgrid(x_grid, y_grid)
            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax2.contour(Xg, Yg, Z, levels=3, colors=[colors[k]],
                        alpha=0.6, linewidths=1)
        except (np.linalg.LinAlgError, ValueError):
            pass
    ax2.set_title(f't-SNE 投影 — 聚类密度等高线', fontsize=12)
    ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')

    plt.tight_layout()
    p = PLOT_DIR / "cluster_tsne.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_cluster_sizes(labels):
    """聚类大小饼图。"""
    unique, counts = np.unique(labels, return_counts=True)
    n = len(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.Set2
    colors = [cmap(i % 8) for i in range(len(unique))]

    # 饼图
    wedges, texts, autotexts = ax1.pie(
        counts, labels=[f'C{k}' for k in unique],
        colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.6)
    for at in autotexts:
        at.set_fontsize(10)
    ax1.set_title(f'聚类比例分布 (n={n})', fontsize=12)

    # 柱状图
    bars = ax2.bar([f'C{k}' for k in unique], counts, color=colors,
                   edgecolor='white')
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + n*0.005,
                 f'{count}\n({count/n*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Epochs'); ax2.set_title('聚类大小', fontsize=12)
    ax2.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    p = PLOT_DIR / "cluster_sizes.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_cluster_profiles(profiles, feature_cols):
    """聚类判别特征热力图 (Top-15 特征)。"""
    n_clusters = len(profiles)

    # 收集所有 top 特征名
    top_set = set()
    for k in profiles:
        for name, _ in profiles[k]["top_features"][:15]:
            top_set.add(name)
    top_features = sorted(top_set)

    if len(top_features) == 0:
        print("  [SKIP] cluster_profiles: 无特征")
        return

    # 构建热力图矩阵
    n_feats = len(top_features)
    heatmap = np.zeros((n_clusters, n_feats))
    feat_to_idx = {f: i for i, f in enumerate(top_features)}

    # 全局 z-score
    for k in profiles:
        center = profiles[k]["center"]
        for feat, idx in feat_to_idx.items():
            feat_idx = feature_cols.index(feat) if feat in feature_cols else -1
            if feat_idx >= 0:
                heatmap[k, idx] = center[feat_idx]

    # 对热力图做 min-max 归一化 (按列, 便于比较)
    heatmap_norm = np.zeros_like(heatmap)
    for j in range(n_feats):
        col_min = heatmap[:, j].min()
        col_max = heatmap[:, j].max()
        if col_max - col_min > 1e-8:
            heatmap_norm[:, j] = (heatmap[:, j] - col_min) / (col_max - col_min)
        else:
            heatmap_norm[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(max(14, n_feats * 0.3),
                                    max(4, n_clusters * 0.8)))
    im = ax.imshow(heatmap_norm, aspect='auto', cmap='RdBu_r',
                   vmin=0, vmax=1)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(top_features, rotation=90, fontsize=6,
                       ha='center')
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f'C{k} ({profiles[k]["pct"]:.0f}%)'
                        for k in range(n_clusters)], fontsize=10)
    ax.set_title('聚类中心特征分布 (min-max 归一化, 红=高)', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8, label='相对强度')

    plt.tight_layout()
    p = PLOT_DIR / "cluster_profiles_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_cluster_by_night(df, labels):
    """绘制每夜聚类分布 (堆叠柱状图)。"""
    df = df.copy()
    df['cluster'] = labels

    if 'night' not in df.columns:
        print("  [SKIP] cluster_by_night: 无 night 列")
        return

    nights = sorted(df['night'].unique())
    n_nights = len(nights)
    n_clusters = len(np.unique(labels))

    # 构建矩阵
    matrix = np.zeros((n_nights, n_clusters))
    for i, night in enumerate(nights):
        night_labels = df[df['night'] == night]['cluster'].values
        for k in range(n_clusters):
            matrix[i, k] = (night_labels == k).sum() / len(night_labels) * 100

    fig, ax = plt.subplots(figsize=(max(14, n_nights * 0.3), 6))
    cmap = plt.cm.Set2
    colors = [cmap(i % 8) for i in range(n_clusters)]

    bottom = np.zeros(n_nights)
    for k in range(n_clusters):
        bars = ax.bar(range(n_nights), matrix[:, k], bottom=bottom,
                      color=colors[k], label=f'C{k}', width=0.8,
                      edgecolor='white', lw=0.3)
        bottom += matrix[:, k]

    ax.set_xticks(range(n_nights))
    ax.set_xticklabels([f'N{n}' for n in nights],
                       fontsize=7, rotation=90, ha='center')
    ax.set_ylabel('%'); ax.set_xlabel('Night')
    ax.set_title('各夜聚类分布', fontsize=12)
    ax.legend(loc='upper right', ncol=n_clusters, fontsize=8, frameon=False)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    p = PLOT_DIR / "cluster_by_night.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════
#  报告和输出
# ═══════════════════════════════════════════════════════════

def save_results(df, labels, model, profiles, feature_cols, qc_summary):
    """保存聚类结果到文件。"""
    # 1. 聚类标签追加到原始数据
    df_out = df.copy()
    df_out['cluster'] = labels
    cluster_csv = OUTPUT_DIR / "clustered_features.csv"
    df_out.to_csv(cluster_csv, index=False)
    print(f"  Saved: {cluster_csv} ({len(df_out)} rows, {len(df_out.columns)} cols)")

    # 2. 聚类中心
    centers_df = pd.DataFrame(
        model.cluster_centers_,
        columns=feature_cols,
        index=[f'C{k}' for k in range(model.cluster_centers_.shape[0])]
    )
    centers_csv = OUTPUT_DIR / "cluster_centers.csv"
    centers_df.to_csv(centers_csv)
    print(f"  Saved: {centers_csv}")

    # 3. 详细报告
    report_path = OUTPUT_DIR / "cluster_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("EEG 特征无监督聚类报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("─── QC 摘要 ───\n")
        for k, v in qc_summary.items():
            f.write(f"  {k}: {v}\n")

        f.write("\n─── 聚类评估 ───\n")
        inertia = model.inertia_
        f.write(f"  算法: KMeans k={N_CLUSTERS}\n")
        f.write(f"  Inertia: {inertia:.1f}\n")

        from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                       calinski_harabasz_score)
        # 只对子集计算以避免内存问题 (大样本)
        n = len(labels)
        if n > 10000:
            rng = np.random.RandomState(RANDOM_STATE)
            sub_idx = rng.choice(n, 10000, replace=False)
            X_sub = model.cluster_centers_[labels[sub_idx]]
            labels_sub = labels[sub_idx]
        else:
            X_sub = None
            labels_sub = labels

        f.write(f"  Silhouette:       {silhouette_score(X_sub if X_sub is not None else 'FULL', labels_sub):.4f}\n")
        f.write(f"  Davies-Bouldin:   {davies_bouldin_score(X_sub if X_sub is not None else 'FULL', labels_sub):.4f}\n")
        f.write(f"  Calinski-Harabasz: {calinski_harabasz_score(X_sub if X_sub is not None else 'FULL', labels_sub):.1f}\n")

        f.write("\n─── 聚类大小 ───\n")
        for k in sorted(profiles.keys()):
            f.write(f"  C{k}: {profiles[k]['size']} epochs "
                    f"({profiles[k]['pct']:.1f}%)\n")

        f.write("\n─── 每类 Top-10 判别特征 ───\n")
        for k in sorted(profiles.keys()):
            f.write(f"\n  C{k} ({profiles[k]['size']} epochs, "
                    f"{profiles[k]['pct']:.1f}%):\n")
            for name, z in profiles[k]['top_features'][:10]:
                direction = "↑" if z > 0 else "↓"
                f.write(f"    {direction} {name:50s} z={z:+.3f}\n")

    print(f"  Saved: {report_path}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main(data_dir=None):
    print("=" * 60)
    print("EEG 特征质量控制 + 归一化 + 无监督聚类 (k=5)")
    print("=" * 60)

    # ── 1. 加载 ──
    print("\n[步骤 1/5] 加载所有夜晚特征数据...")
    df_all, meta_list = load_all_features(data_dir)

    # ── 2. QC ──
    print("\n[步骤 2/5] 质量控制...")
    meta_cols = [c for c in df_all.columns if c in META_COLUMNS]
    feature_cols = [c for c in df_all.columns
                    if c not in META_COLUMNS
                    and df_all[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    qc_report(df_all, feature_cols)

    df_clean, feature_cols, qc_summary = apply_qc(df_all)

    # ── 3. 归一化 ──
    print("\n[步骤 3/5] 归一化 (RobustScaler)...")
    X_scaled, scaler, feature_cols = normalize_features(df_clean, feature_cols)

    # ── 4. 聚类 ──
    print("\n[步骤 4/5] KMeans 聚类 (k=5)...")
    labels, model, cluster_metrics = run_clustering(X_scaled)
    profiles = characterize_clusters(X_scaled, labels, feature_cols, model)

    # ── 5. 可视化 + 输出 ──
    print("\n[步骤 5/5] 可视化与输出...")

    # 提取 meta 部分用于可视化
    df_meta = df_clean[[c for c in df_clean.columns
                        if c in META_COLUMNS]].copy()

    plot_tsne(X_scaled, labels)
    plot_cluster_sizes(labels)
    plot_cluster_profiles(profiles, feature_cols)
    plot_cluster_by_night(df_clean, labels)

    # 保存结果
    save_results(df_clean, labels, model, profiles, feature_cols, qc_summary)

    print("\n" + "=" * 60)
    print("分析完成! 输出文件:")
    print(f"  {OUTPUT_DIR / 'clustered_features.csv'}")
    print(f"  {OUTPUT_DIR / 'cluster_centers.csv'}")
    print(f"  {OUTPUT_DIR / 'cluster_report.txt'}")
    print(f"  图表: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EEG 特征 QC + 聚类")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="特征文件目录 (默认: 当前脚本目录)")
    args = parser.parse_args()
    main(data_dir=args.data_dir)
