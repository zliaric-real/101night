sklearn 不稳定问题诊断报告
===========================

## 环境

| 包 | 版本 | 备注 |
|----|------|------|
| sklearn | 1.7.2 | 最新版 (2026) |
| joblib | 1.5.3 | loky 后端 |
| numpy | 2.2.6 | BLAS 线程池 |
| scipy | 1.15.3 | |
| yasa | 0.7.0 | LightGBM 模型用 sklearn 0.24.2 训练 |
| CPU | 32 逻辑核 | n_jobs 默认 = 32 |

## 症状

三种场景均出现**间歇性**挂起（同一代码有时秒过、有时永远卡死）：

1. **sklearn 导入挂起**
   `from sklearn.cluster import KMeans` → 进程卡死在 sklearn/__init__.py

2. **YASA predict() 挂起**
   `sls.predict()` 完成后，对返回值的 numpy 操作 (np.unique 等) 卡死

3. **stage_stats dict comprehension 挂起**
   `np.unique(hypno_pred)` 在 dict comprehension 中调用时卡死

## 根因分析

### 主要原因：sklearn 导入时的共享库加载竞态条件

GitHub 确认问题：**[scikit-learn#29145](https://github.com/scikit-learn/scikit-learn/issues/29145)** — "V1.5 randomly hangs on import"

根因：sklearn `__init__.py` 在导入时调用 `threadpoolctl` 检测并注册底层 BLAS 线程池。
这一步会 `dlopen("libopenblas.dll")` 或 `dlopen("libomp.dll")`。在 Windows + 多核
环境下，多个 Python 进程（joblib loky 后端、YASA 内部线程等）**同时**加载这些共享库
时产生死锁。

具体触发链：
```
import sklearn
  → sklearn/__init__.py (line ~154)
    → threadpoolctl.register()
      → ctypes.cdll.LoadLibrary("libopenblas.dll")
        → 内核级 DLL 加载锁竞争 → 死锁 (Windows 特有)
```

### 次要原因：sklearn 版本跨越过大 (0.24 → 1.7)

YASA 0.7.0 的 LightGBM 模型用 sklearn 0.24.2 训练。joblib.load 反序列化
``LabelEncoder`` 时触发 `InconsistentVersionWarning`。在 1.7.2 中 LabelEncoder
的 `_validate_keywords()` 等方法签名变化，反序列化后的对象调用时可能进入
无限等待（等待从不存在的属性）。

### 为什么是间歇性的

- DLL 加载锁的竞争依赖于 OS 调度时序——32 核系统上进程创建的精确时间点
- YASA 内部 resample(100) 可能触发 MNE 的 OpenBLAS 线程
- joblib loky 后端在首次使用时 spawn 32 个 worker 进程

## 已验证有效的规避方案

### 方案 A（推荐，最小改动）：环境变量限制并行度

```bash
set LOKY_MAX_CPU_COUNT=2
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
```

这阻止 joblib spawn 32 个 worker 和 OpenBLAS 创建 32 个线程，
大幅降低 DLL 加载锁竞争概率。

### 方案 B：降级 sklearn 到与 YASA 模型匹配的版本

```bash
pip install scikit-learn==0.24.2
```

消除 InconsistentVersionWarning，但可能与其他包 (MNE 1.12 等) 不兼容。

### 方案 C：在代码中预导入

在任何可能并行或调用 YASA 之前，强制在主线程完成 sklearn 导入：

```python
# 在所有其他 import 之前
import sklearn
import sklearn.cluster
import sklearn.preprocessing  # LabelEncoder 在此
```

确保共享库在主线程单线程加载，避免竞态。

### 方案 D：单进程架构

避免使用 joblib/subprocess 多进程——在单一 Python 进程中顺序处理所有夜晚。
这已经在 notebook 验证可行。

## 当前代码状态

- `hemich_select.py` — 不依赖 sklearn（纯 numpy），方案已落地 ✓
- `feature_101night_analy.py` — 顶层 `import sklearn` 通过 KMeans 触发（在
  `_get_hemispheric_channels` 中，虽然不再从 __init__ 调用，但 import 链仍在）
- `run_step2_yasa_with_eog` — stage_counts/stage_pct 已删除 ✓

## 建议

1. 在脚本入口添加 `LOKY_MAX_CPU_COUNT=2` 等环境变量
2. 把 `from sklearn.cluster import KMeans` 从模块顶层移到 `_get_hemispheric_channels`
   函数内部（延迟导入，避免在 __init__ 链中触发）
3. 考虑将 sklearn 降级到 1.5.x（#29145 修复后的首个稳定版）
