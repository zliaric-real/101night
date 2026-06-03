# -*- coding: utf-8 -*-
"""
cross_night_autocorr.py — 跨夜时间序列自相关分析
================================================
对 101 天睡眠指标进行时间序列自相关分析，探索睡眠结构的日间依赖性和周期性。

分析内容:
  1. 睡眠时长 (hours) 的自相关
  2. 睡眠效率 (%) 的自相关
  3. 各睡眠阶段比例 (Wake/N1/N2/N3/REM) 的自相关
  4. N3+REM 深度睡眠指标的自相关
  5. 交叉相关矩阵 (各指标之间的时滞相关性)
  6. 周周期检测 (lag-7 自相关显著性)
  7. 滑动窗自相关 — 检测随时间变化的自相关结构

输入:
  - batch_results.pkl: 批量 YASA 分期结果 (需先运行 batch_analyze.py 至完成)

输出:
  - sleep_plots/autocorr_duration.png
  - sleep_plots/autocorr_efficiency.png
  - sleep_plots/autocorr_stages.png
  - sleep_plots/autocorr_cross_correlation.png
  - sleep_plots/autocorr_weekly_rhythm.png
  - sleep_plots/autocorr_rolling.png
  - autocorr_report.csv: 汇总统计表

用法:
  conda activate eeg_101night
  python cross_night_autocorr.py
"""

import pickle
import warnings
from pathlib import Path

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
CACHE_PATH = OUTPUT_DIR / "batch_results.pkl"
REPORT_PATH = OUTPUT_DIR / "autocorr_report.csv"

# ── 配置 ──
MAX_LAG = 30          # 最大滞后天数
ALPHA = 0.05          # 显著性水平
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {"Wake": "#fc8d62", "N1": "#66c2a5", "N2": "#8da0cb",
                "N3": "#a6d854", "REM": "#e78ac3"}


# ═══════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════

def load_nightly_metrics():
    """从 batch_results.pkl 加载每夜的睡眠指标时间序列。

    Returns:
        df: DataFrame, index=night, columns=指标
            包含: duration_h, efficiency_pct,
                  wake_pct, n1_pct, n2_pct, n3_pct, rem_pct,
                  n3_rem_pct, n_epochs
    """
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{CACHE_PATH} 不存在。请先运行 batch_analyze.py 完成 YASA 分期。"
        )

    with open(CACHE_PATH, "rb") as f:
        results = pickle.load(f)

    # 只保留 YASA 成功的夜
    valid = [r for r in results if r.get("yasa_ok") and r.get("stage_counts")]
    if not valid:
        raise ValueError("没有成功的 YASA 分期结果。请先运行 batch_analyze.py。")

    records = []
    for r in valid:
        ct = r["stage_counts"]
        total = sum(ct.values())
        n_epochs = r.get("n_epochs", total)
        duration_h = n_epochs * 30 / 3600

        records.append({
            "night": r["night"],
            "date": r["date"],
            "duration_h": duration_h,
            "efficiency_pct": (total - ct.get("Wake", 0)) / total * 100,
            "wake_pct": ct.get("Wake", 0) / total * 100,
            "n1_pct": ct.get("N1", 0) / total * 100,
            "n2_pct": ct.get("N2", 0) / total * 100,
            "n3_pct": ct.get("N3", 0) / total * 100,
            "rem_pct": ct.get("REM", 0) / total * 100,
            "n3_rem_pct": (ct.get("N3", 0) + ct.get("REM", 0)) / total * 100,
            "n_epochs": n_epochs,
        })

    df = pd.DataFrame(records).sort_values("night").set_index("night")
    print(f"加载 {len(df)} 夜有效数据 (共 {len(results)} 条记录)")
    print(f"日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    return df


# ═══════════════════════════════════════════════════════════
#  自相关计算
# ═══════════════════════════════════════════════════════════

def compute_acf(series, nlags=MAX_LAG):
    """计算自相关函数 (ACF) 及置信区间。

    Returns:
        dict: lags, acf, ci_upper, ci_lower, is_significant
    """
    n = len(series)
    if n < 10:
        return None

    # 去均值
    x = series.values - series.mean()
    var = np.sum(x ** 2) / n

    acf_vals = np.zeros(nlags + 1)
    for lag in range(nlags + 1):
        if lag == 0:
            acf_vals[lag] = 1.0
        else:
            acf_vals[lag] = np.sum(x[lag:] * x[:-lag]) / (var * n)

    # Bartlett 标准误 (1/sqrt(n))
    se = 1.0 / np.sqrt(n)
    z = stats.norm.ppf(1 - ALPHA / 2)  # 双尾
    ci = z * se

    return {
        "lags": np.arange(nlags + 1),
        "acf": acf_vals,
        "ci_upper": np.full(nlags + 1, ci),
        "ci_lower": np.full(nlags + 1, -ci),
        "significant": np.abs(acf_vals) > ci,
    }


def compute_pacf(series, nlags=MAX_LAG):
    """计算偏自相关函数 (PACF) — 使用 Levinson-Durbin 递推或直接回归。

    由于 scipy 无内置 PACF，使用 statsmodels 的 yule_walker 逐阶计算。
    如果 statsmodels 不可用，回退到 OLS 回归法。
    """
    try:
        from statsmodels.tsa.stattools import pacf
        pacf_vals = pacf(series.values, nlags=nlags, method='ywm')
        return pacf_vals
    except ImportError:
        pass

    # 回退: OLS 法 (慢但可靠)
    n = len(series)
    x = series.values - series.mean()
    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0

    for lag in range(1, nlags + 1):
        if n <= lag + 5:
            pacf_vals[lag] = np.nan
            continue
        # 用 lag 个历史值预测当前值
        y = x[lag:]
        X = np.column_stack([x[lag - i - 1: -i - 1 if i > 0 else None]
                             for i in range(lag)])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            pacf_vals[lag] = beta[-1]
        except np.linalg.LinAlgError:
            pacf_vals[lag] = np.nan

    return pacf_vals


def compute_cross_correlation(x, y, nlags=MAX_LAG):
    """计算交叉相关函数 (CCF)。

    Returns:
        lags, ccf: 正滞后表示 x 领先 y
    """
    n = len(x)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    ccf_vals = np.zeros(2 * nlags + 1)
    lags = np.arange(-nlags, nlags + 1)

    for i, lag in enumerate(lags):
        if lag < 0:
            ccf_vals[i] = np.corrcoef(x[-lag:], y[:lag])[0, 1] if -lag < n else np.nan
        elif lag > 0:
            ccf_vals[i] = np.corrcoef(x[:-lag], y[lag:])[0, 1] if lag < n else np.nan
        else:
            ccf_vals[i] = np.corrcoef(x, y)[0, 1]

    return lags, ccf_vals


# ═══════════════════════════════════════════════════════════
#  Week周期检测
# ═══════════════════════════════════════════════════════════

def test_weekly_rhythm(series):
    """检测 lag-7 自相关是否显著（周周期）。

    零假设: 无周周期 (使用随机排列检验)
    """
    n = len(series)
    if n < 14:
        return {"lag7_acf": np.nan, "p_value": np.nan, "significant": False}

    acf_result = compute_acf(series, nlags=7)
    lag7_acf = acf_result["acf"][7]

    # 排列检验 (1000次)
    n_perm = 1000
    perm_acfs = np.zeros(n_perm)
    rng = np.random.RandomState(42)

    for i in range(n_perm):
        permuted = rng.permutation(series.values)
        x = permuted - permuted.mean()
        var = np.sum(x ** 2) / n
        perm_acfs[i] = np.sum(x[7:] * x[:-7]) / (var * n)

    p_value = (np.sum(np.abs(perm_acfs) >= np.abs(lag7_acf)) + 1) / (n_perm + 1)

    return {
        "lag7_acf": lag7_acf,
        "p_value": p_value,
        "significant": p_value < ALPHA,
    }


# ═══════════════════════════════════════════════════════════
#  滚动窗自相关
# ═══════════════════════════════════════════════════════════

def rolling_autocorr(series, window=20, lags=(1, 3, 7)):
    """计算滑动窗内的自相关，检测时间变化。

    Args:
        series: 时间序列
        window: 滑动窗口大小 (夜数)
        lags: 关注的自相关滞后

    Returns:
        DataFrame: index=窗口中心night, columns=lag_1, lag_3, lag_7
    """
    n = len(series)
    if n < window:
        return None

    results = []
    for i in range(n - window + 1):
        win = series.iloc[i:i + window]
        row = {"night": series.index[i + window // 2]}
        for lag in lags:
            if lag < window:
                row[f"lag_{lag}"] = np.corrcoef(win.iloc[lag:], win.iloc[:-lag])[0, 1]
            else:
                row[f"lag_{lag}"] = np.nan
        results.append(row)

    return pd.DataFrame(results).set_index("night")


# ═══════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════

def plot_single_acf(series, name, title, color="#2c7bb6", ylabel="ACF"):
    """绘制单个指标的自相关图 (ACF + PACF)。"""
    acf_r = compute_acf(series, MAX_LAG)
    pacf_vals = compute_pacf(series, MAX_LAG)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ACF
    ax1.stem(acf_r["lags"], acf_r["acf"], linefmt=color, markerfmt='o',
             basefmt='k-')
    ax1.axhline(y=0, color='black', lw=0.5)
    ax1.fill_between(acf_r["lags"], acf_r["ci_upper"], acf_r["ci_lower"],
                     alpha=0.15, color='gray')
    ax1.axhline(y=acf_r["ci_upper"][0], color='gray', ls='--', lw=1,
                label=f'{ALPHA*100:.0f}% 置信界')
    ax1.axhline(y=-acf_r["ci_upper"][0], color='gray', ls='--', lw=1)
    sig_lags = acf_r["lags"][acf_r["significant"]]
    sig_acfs = acf_r["acf"][acf_r["significant"]]
    ax1.scatter(sig_lags, sig_acfs, color='red', s=30, zorder=5, label='显著')
    ax1.set_xlabel('滞后 (天)'); ax1.set_ylabel(ylabel)
    ax1.set_title(f'{title} — ACF'); ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # PACF
    ax2.stem(range(len(pacf_vals)), pacf_vals, linefmt='#d95f02', markerfmt='o',
             basefmt='k-')
    ax2.axhline(y=0, color='black', lw=0.5)
    se = 1.0 / np.sqrt(len(series))
    z = stats.norm.ppf(1 - ALPHA / 2)
    ax2.axhline(y=z*se, color='gray', ls='--', lw=1, label=f'{ALPHA*100:.0f}% 置信界')
    ax2.axhline(y=-z*se, color='gray', ls='--', lw=1)
    ax2.set_xlabel('滞后 (天)'); ax2.set_ylabel('PACF')
    ax2.set_title(f'{title} — PACF'); ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = PLOT_DIR / f"autocorr_{name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_stage_autocorr_grid(df):
    """绘制所有睡眠阶段的自相关网格图。"""
    stages = ["wake_pct", "n1_pct", "n2_pct", "n3_pct", "rem_pct"]
    labels = ["Wake", "N1", "N2", "N3", "REM"]
    colors = [STAGE_COLORS[s] for s in labels]

    fig, axes = plt.subplots(len(stages), 1, figsize=(14, 2.5 * len(stages)),
                             sharex=True)
    fig.suptitle("睡眠阶段占比 — 自相关函数 (ACF)", fontsize=14, y=1.01)

    for ax, col, label, color in zip(axes, stages, labels, colors):
        acf_r = compute_acf(df[col], MAX_LAG)
        ax.stem(acf_r["lags"], acf_r["acf"], linefmt=color, markerfmt='o',
                basefmt='k-')
        ax.axhline(y=0, color='black', lw=0.5)
        ax.fill_between(acf_r["lags"], acf_r["ci_upper"], acf_r["ci_lower"],
                        alpha=0.15, color='gray')
        ax.axhline(y=acf_r["ci_upper"][0], color='gray', ls='--', lw=0.8)
        ax.axhline(y=-acf_r["ci_upper"][0], color='gray', ls='--', lw=0.8)
        ax.set_ylabel(f'{label} ACF'); ax.grid(True, alpha=0.3)
        # 标注显著滞后
        sig_lags = acf_r["lags"][acf_r["significant"] & (acf_r["lags"] > 0)]
        for lag in sig_lags[:5]:  # 最多标5个
            ax.annotate(f'lag{lag}', (lag, acf_r["acf"][lag]),
                        fontsize=7, color='red')

    axes[-1].set_xlabel('滞后 (天)')
    plt.tight_layout()
    p = PLOT_DIR / "autocorr_stages.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_cross_correlation_matrix(df):
    """绘制各睡眠指标间的交叉相关矩阵。"""
    metrics = ["duration_h", "efficiency_pct",
               "wake_pct", "n1_pct", "n2_pct", "n3_pct", "rem_pct"]
    labels = ["Duration", "Efficiency", "Wake", "N1", "N2", "N3", "REM"]
    n = len(metrics)

    fig, axes = plt.subplots(n, n, figsize=(16, 14))
    fig.suptitle("睡眠指标交叉相关矩阵 (CCF)", fontsize=14, y=1.01)

    for i, (mi, li) in enumerate(zip(metrics, labels)):
        for j, (mj, lj) in enumerate(zip(metrics, labels)):
            ax = axes[i][j]
            if i == j:
                # 对角线: 自相关
                acf_r = compute_acf(df[mi], MAX_LAG)
                ax.stem(acf_r["lags"], acf_r["acf"], linefmt='#2c7bb6',
                        markerfmt='.', basefmt='k-')
                ax.axhline(y=0, color='black', lw=0.3)
                ax.set_title(f'{li} ACF', fontsize=9)
            else:
                # 非对角线: 交叉相关
                lags, ccf = compute_cross_correlation(df[mi], df[mj], 10)
                colors_ccf = ['#d95f02' if v > 0 else '#7570b3' for v in ccf]
                ax.bar(lags, ccf, color=colors_ccf, width=0.6, alpha=0.7)
                ax.axhline(y=0, color='black', lw=0.3)
                # 标注最大交叉相关
                max_idx = np.argmax(np.abs(ccf))
                ax.annotate(f'lag={lags[max_idx]}\nr={ccf[max_idx]:.2f}',
                            (lags[max_idx], ccf[max_idx]),
                            fontsize=7, ha='center', va='bottom')
                if i == 0:
                    ax.set_title(f'{li} → {lj}', fontsize=9)
            if i == n - 1:
                ax.set_xlabel('滞后 (天)', fontsize=7)
            if j == 0:
                ax.set_ylabel(li, fontsize=8)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = PLOT_DIR / "autocorr_cross_correlation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_weekly_rhythm(df):
    """绘制周周期检测结果。"""
    metrics = ["duration_h", "efficiency_pct", "n3_pct", "rem_pct"]
    labels = ["Sleep Duration", "Sleep Efficiency", "N3 (Deep)", "REM"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("周周期检测 (lag-7 自相关显著性)", fontsize=14, y=1.01)

    for ax, col, label in zip(axes.flat, metrics, labels):
        acf_r = compute_acf(df[col], 14)
        # 只显示 lag 1-14
        lags = acf_r["lags"][1:]
        acf_vals = acf_r["acf"][1:]
        sig = acf_r["significant"][1:]

        colors = ['#e41a1c' if (s and lag == 7) else
                  '#377eb8' if s else '#cccccc'
                  for s, lag in zip(sig, lags)]
        ax.bar(lags, acf_vals, color=colors, width=0.5, edgecolor='white')

        ax.axhline(y=0, color='black', lw=0.5)
        ax.axhline(y=acf_r["ci_upper"][0], color='gray', ls='--', lw=1)
        ax.axhline(y=-acf_r["ci_upper"][0], color='gray', ls='--', lw=1)

        # 标注 lag-7
        if len(acf_vals) >= 7:
            week_acf = acf_vals[6]  # index 6 = lag 7
            ax.annotate(f'lag-7 = {week_acf:.3f}',
                        (7, week_acf), fontsize=10, fontweight='bold',
                        color='#e41a1c' if sig[6] else 'gray',
                        xytext=(7, week_acf + 0.05),
                        ha='center')

        ax.set_title(label, fontsize=11)
        ax.set_xlabel('滞后 (天)'); ax.set_ylabel('ACF')
        ax.set_xticks(range(1, 15))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = PLOT_DIR / "autocorr_weekly_rhythm.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def plot_rolling_autocorr(df):
    """绘制滑动窗自相关变化。"""
    metrics = ["duration_h", "efficiency_pct", "n3_pct", "rem_pct"]
    labels = ["Sleep Duration", "Sleep Efficiency", "N3 (Deep)", "REM"]

    # 对每个指标做滚动自相关
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3 * len(metrics)),
                             sharex=True)
    fig.suptitle("滑动窗自相关 (窗宽=20夜)", fontsize=14, y=1.01)

    for ax, col, label in zip(axes, metrics, labels):
        roll = rolling_autocorr(df[col], window=20, lags=(1, 3, 7))
        if roll is None:
            continue
        ax.plot(roll.index, roll["lag_1"], 'o-', color='#377eb8',
                markersize=4, lw=1.5, label='Lag-1 (日间)')
        ax.plot(roll.index, roll["lag_3"], 's-', color='#4daf4a',
                markersize=4, lw=1.5, label='Lag-3')
        ax.plot(roll.index, roll["lag_7"], '^-', color='#e41a1c',
                markersize=5, lw=1.5, label='Lag-7 (周)')
        ax.axhline(y=0, color='black', lw=0.5, ls=':')
        ax.set_ylabel(f'{label} ACF'); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('夜编号')
    plt.tight_layout()
    p = PLOT_DIR / "autocorr_rolling.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════

def generate_report(df):
    """生成自相关汇总报告 CSV。"""
    metrics = ["duration_h", "efficiency_pct",
               "wake_pct", "n1_pct", "n2_pct", "n3_pct", "rem_pct"]
    labels = ["Duration", "Efficiency", "Wake", "N1", "N2", "N3", "REM"]
    key_lags = [1, 3, 7, 14]

    rows = []
    for col, label in zip(metrics, labels):
        acf_r = compute_acf(df[col], max(key_lags))
        pacf_vals = compute_pacf(df[col], max(key_lags))
        weekly = test_weekly_rhythm(df[col])

        row = {"metric": label, "n_nights": len(df[col].dropna())}
        row.update({f"mean": df[col].mean(), f"std": df[col].std()})

        for lag in key_lags:
            if lag < len(acf_r["acf"]):
                row[f"ACF_lag{lag}"] = acf_r["acf"][lag]
                row[f"ACF_lag{lag}_sig"] = acf_r["significant"][lag]
                row[f"PACF_lag{lag}"] = pacf_vals[lag] if lag < len(pacf_vals) else np.nan
        row["weekly_p_value"] = weekly["p_value"]
        row["weekly_significant"] = weekly["significant"]
        rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)
    print(f"\n  Report saved: {REPORT_PATH}")
    print(report.round(3).to_string())


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("跨夜自相关分析 — 101-nights 睡眠结构时间依赖性")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载每夜睡眠指标...")
    df = load_nightly_metrics()
    print(f"  可用指标: {list(df.columns)}")
    print(f"  描述统计:\n{df.describe().round(1).to_string()}")

    # 2. 单项自相关图
    print("\n[2/6] 睡眠时长 & 效率自相关...")
    plot_single_acf(df["duration_h"], "duration",
                    "睡眠时长 (小时)", color="#2c7bb6")
    plot_single_acf(df["efficiency_pct"], "efficiency",
                    "睡眠效率 (%)", color="#e78ac3")

    # 3. 阶段自相关网格
    print("\n[3/6] 睡眠阶段自相关网格...")
    plot_stage_autocorr_grid(df)

    # 4. 交叉相关矩阵
    print("\n[4/6] 交叉相关矩阵...")
    plot_cross_correlation_matrix(df)

    # 5. 周周期检测
    print("\n[5/6] 周周期 (lag-7) 检测...")
    plot_weekly_rhythm(df)

    # 测试并报告
    print("\n  周周期显著性 (排列检验):")
    for col, label in [("duration_h", "睡眠时长"),
                        ("efficiency_pct", "睡眠效率"),
                        ("n3_pct", "N3 深睡"),
                        ("rem_pct", "REM")]:
        weekly = test_weekly_rhythm(df[col])
        sig = "*** 显著" if weekly["significant"] else "不显著"
        print(f"    {label:10s}: lag-7 ACF={weekly['lag7_acf']:.3f}, "
              f"p={weekly['p_value']:.3f} [{sig}]")

    # 6. 滑动窗自相关
    print("\n[6/6] 滑动窗自相关...")
    plot_rolling_autocorr(df)

    # 报告
    print("\n生成汇总报告...")
    generate_report(df)

    print("\n" + "=" * 60)
    print("分析完成! 图表已保存到:", PLOT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
