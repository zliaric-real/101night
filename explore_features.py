# -*- coding: utf-8 -*-
"""特征探索分析 — 101-nights Night 040"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

OUTDIR = Path("E:/idea/101night/sleep_plots")
OUTDIR.mkdir(exist_ok=True)

# ── 1. 加载数据 ──────────────────────────────────────────
print("=" * 60)
print("1. 加载特征数据")
print("=" * 60)

df = pd.read_csv("E:/idea/101night/features_night40_final.csv")
print(f"  形状: {df.shape[0]} epochs × {df.shape[1]} 特征列")
print(f"  epoch 范围: {df['epoch'].min()} – {df['epoch'].max()}")
print(f"  总时长: {df.shape[0] * 30 / 3600:.1f} 小时 (按30s/epoch估算)")

# ── 2. 基础统计 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 基础统计概览")
print("=" * 60)

# 数值列
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'epoch']

# 缺失率
missing = df[num_cols].isnull().mean().sort_values(ascending=False)
print(f"\n  特征总数(数值): {len(num_cols)}")
print(f"  完全缺失的特征: {sum(missing == 1.0)}")
print(f"  有缺失的特征: {sum(missing > 0)}")

bad_cols = missing[missing > 0.5].index.tolist()
good_cols = [c for c in num_cols if c not in bad_cols]
print(f"  可用特征(缺失<50%): {len(good_cols)}")
print(f"  高缺失特征: {bad_cols}")

# ── 3. 按特征组分类 ──────────────────────────────────────
print("\n" + "=" * 60)
print("3. 特征组概览")
print("=" * 60)

groups = {
    'spectral': [c for c in num_cols if 'spectral' in c or c.startswith('aperiodic')],
    'hjorth': [c for c in num_cols if 'hjorth' in c],
    'statistical': [c for c in num_cols if 'statistical' in c],
    'entropy': [c for c in num_cols if 'entropy' in c and 'spectral' not in c],
    'complexity': [c for c in num_cols if any(x in c for x in ['lz_', 'higuchi', 'dfa'])],
    'catch22': [c for c in num_cols if 'catch22' in c],
    'wavelet': [c for c in num_cols if 'wavelet' in c],
    'tsfresh': [c for c in num_cols if 'tsfresh' in c],
    'rqa': [c for c in num_cols if 'rqa' in c],
    'adv_spectral': [c for c in num_cols if 'adv_spectral' in c],
    'autocorrelation': [c for c in num_cols if 'autocorrelation' in c],
    'connectivity': [c for c in num_cols if 'connectivity' in c or 'gfp' in c or 'omega' in c],
    'graph': [c for c in num_cols if 'graph' in c],
}

for name, cols in groups.items():
    available = sum(1 for c in cols if c in good_cols)
    print(f"  {name:20s}: {available:3d}/{len(cols):3d} 可用")

# ── 4. 关键特征的分布 ────────────────────────────────────
print("\n" + "=" * 60)
print("4. 关键特征分布 (前15个)")
print("=" * 60)

key_features = ['spectral_entropy', 'aperiodic_slope', 'sample_entropy', 
                'higuchi_fd', 'dfa', 'catch22_0', 'wavelet_delta',
                'wavelet_theta', 'rqa_RR', 'rqa_DET']

for feat in key_features:
    if feat in df.columns:
        vals = df[feat].dropna()
        if len(vals) > 0:
            print(f"  {feat:25s}: mean={vals.mean():8.4f}  std={vals.std():8.4f}  "
                  f"min={vals.min():8.4f}  max={vals.max():8.4f}  missing={df[feat].isnull().mean():.1%}")

# ── 5. 特征分布直方图 ────────────────────────────────────
print("\n" + "=" * 60)
print("5. 绘制特征分布图...")
print("=" * 60)

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
plot_feats = [f for f in key_features if f in df.columns][:12]

for ax, feat in zip(axes.flat, plot_feats):
    vals = df[feat].dropna()
    ax.hist(vals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.suptitle('Night 040 — Key Feature Distributions', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUTDIR / "feature_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: {OUTDIR / 'feature_distributions.png'}")

# ── 6. 特征相关性热图 ────────────────────────────────────
print("\n" + "=" * 60)
print("6. 绘制特征相关性热图...")
print("=" * 60)

# 选20个代表性特征
corr_features = [f for f in key_features + 
    ['lz_complexity', 'wavelet_alpha', 'wavelet_beta', 'wavelet_gamma',
     'permutation_entropy', 'rqa_LAM', 'rqa_TT', 'gfp', 
     'adv_spectral_centroid', 'adv_spectral_sef50'] if f in df.columns]

corr_features = list(dict.fromkeys(corr_features))  # dedup
corr_features = [c for c in corr_features if c in good_cols][:20]

corr_df = df[corr_features].dropna()
if len(corr_df) > 100:
    corr_df = corr_df.iloc[::max(1, len(corr_df)//500)]  # subsample if huge

corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, annot=False,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Night 040 — Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(OUTDIR / "feature_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: {OUTDIR / 'feature_correlation.png'}")

# 找高相关对
print("\n  高相关特征对 (|r| > 0.8):")
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.8:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
for a, b, r in high_corr:
    print(f"    {a:30s} ↔ {b:30s}  r={r:+.3f}")

# ── 7. 时间演化趋势 ──────────────────────────────────────
print("\n" + "=" * 60)
print("7. 绘制时间演化趋势...")
print("=" * 60)

fig, axes = plt.subplots(4, 2, figsize=(18, 14))
trend_feats = ['spectral_entropy', 'sample_entropy', 'dfa', 'higuchi_fd',
               'aperiodic_slope', 'wavelet_delta', 'wavelet_theta', 'rqa_DET']
trend_feats = [f for f in trend_feats if f in df.columns]

for ax, feat in zip(axes.flat, trend_feats):
    vals = df[feat].values
    # rolling smooth
    ax.plot(df['epoch'].values, vals, alpha=0.3, linewidth=0.5, color='steelblue')
    if len(vals) > 20:
        window = max(20, len(vals) // 20)
        smooth = pd.Series(vals).rolling(window=window, center=True).mean()
        ax.plot(df['epoch'].values, smooth.values, color='darkred', linewidth=2)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel('Epoch (30s)')

plt.suptitle('Night 040 — Feature Time Evolution', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUTDIR / "feature_trends.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: {OUTDIR / 'feature_trends.png'}")

# ── 8. PCA 降维视图 ──────────────────────────────────────
print("\n" + "=" * 60)
print("8. PCA 降维视图...")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca_cols = [c for c in good_cols if c in corr_features]
if len(pca_cols) < 5:
    pca_cols = good_cols[:20]

pca_df = df[pca_cols].dropna()
if len(pca_df) > 2000:
    pca_df = pca_df.iloc[::max(1, len(pca_df)//2000)]

X = StandardScaler().fit_transform(pca_df)
pca = PCA(n_components=min(10, len(pca_cols)))
X_pca = pca.fit_transform(X)

# 方差解释
print(f"  PCA 使用 {len(pca_cols)} 个特征, {len(pca_df)} 个样本")
for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_[:5], 
                                     np.cumsum(pca.explained_variance_ratio_)[:5])):
    print(f"    PC{i+1}: {var:.1%} (累计 {cum:.1%})")

# PCA 散点图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=np.arange(len(X_pca)), 
                cmap='viridis', alpha=0.6, s=10)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA — Colored by Time (Epoch Order)')
plt.colorbar(sc, ax=ax, label='Epoch')

# Loading plot
ax = axes[1]
loadings = pca.components_[:2].T
for i, feat in enumerate(pca_cols):
    ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
             head_width=0.03, head_length=0.05, fc='steelblue', ec='steelblue', alpha=0.6)
    if abs(loadings[i, 0]) > 0.2 or abs(loadings[i, 1]) > 0.2:
        ax.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feat, fontsize=7)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA Loadings')
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTDIR / "pca_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: {OUTDIR / 'pca_overview.png'}")

print("\n" + "=" * 60)
print("分析完成! 图片保存在: " + str(OUTDIR))
print("=" * 60)
