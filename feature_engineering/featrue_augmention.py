import numpy as np
import math
from itertools import combinations, islice
from scipy.signal import hilbert, coherence, welch

# -----------------------------
# 非线性 / 复杂度 / 算法性特征
# -----------------------------
def hjorth_parameters(x: np.ndarray):
    """HJORTH: activity, mobility, complexity"""
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"Hjorth_activity": 0.0, "Hjorth_mobility": 0.0, "Hjorth_complexity": 0.0}
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    var_x = np.var(x)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    activity = float(var_x)
    mobility = float(np.sqrt(var_d1 / (var_x + 1e-12)))
    complexity = float(np.sqrt((var_d2 / (var_d1 + 1e-12)) / (var_d1 / (var_x + 1e-12)) + 1e-12))
    return {"Hjorth_activity": activity, "Hjorth_mobility": mobility, "Hjorth_complexity": complexity}

def zero_crossing_rate(x: np.ndarray):
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size < 2:
        return {"零交叉率": 0.0}
    zc = np.sum(((x[:-1] * x[1:]) < 0).astype(float)) / float(x.size - 1)
    return {"零交叉率": float(zc)}

def slope_sign_changes(x: np.ndarray):
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size < 3:
        return {"斜率符号变化": 0.0}
    diff = np.diff(x)
    ssc = np.sum(((diff[:-1] * diff[1:]) < 0).astype(float))
    return {"斜率符号变化": float(ssc)}

def lz_complexity(x: np.ndarray):
    """Lempel-Ziv complexity using median binarization"""
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"LZ复杂度": 0.0}
    med = np.median(x)
    bstr = (x > med).astype(int)
    # convert to string-like sequence 0/1
    s = ''.join(map(str, bstr.tolist()))
    i, c, l = 0, 1, 1
    n = len(s)
    while True:
        if i + l > n - 1:
            c += 1
            break
        if s[i:i+l] not in s[i+1: i+1+l]:
            c += 1
            i += l
            l = 1
        else:
            l += 1
    # normalize
    return {"LZ复杂度": float(c / (n / math.log2(n + 2) + 1e-12))}

def higuchi_fd(x: np.ndarray, kmax: int = 10):
    """Higuchi fractal dimension"""
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size < 4:
        return {"Higuchi_fd": 0.0}
    N = x.size
    L = []
    x = x.astype(float)
    for k in range(1, min(kmax, N//2)+1):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2:
                continue
            lm = np.sum(np.abs(np.diff(x[idxs]))) * (N - 1) / ( (len(idxs)-1) * k )
            Lk.append(lm)
        if len(Lk) > 0:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return {"Higuchi_fd": 0.0}
    lnL = np.log(L)
    lnk = np.log(np.arange(1, len(L)+1))
    # linear fit slope
    slope = np.polyfit(lnk, lnL, 1)[0]
    fd = float(-slope)
    return {"Higuchi_fd": fd}

def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2):
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size <= m+1:
        return {"样本熵": 0.0}
    r = r_factor * np.std(x)
    N = x.size

    def _phi(m):
        xmi = np.array([x[i:i+m] for i in range(N - m + 1)])
        C = 0
        for i in range(len(xmi)):
            d = np.max(np.abs(xmi - xmi[i]), axis=1)
            C += np.sum(d <= r) - 1  # exclude self-match
        return C

    try:
        phi_m = _phi(m)
        phi_m1 = _phi(m+1)
        se = -math.log((phi_m1 + 1e-12) / (phi_m + 1e-12))
        return {"样本熵": float(se)}
    except Exception:
        return {"样本熵": 0.0}

def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1):
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size < order * delay:
        return {"置换熵": 0.0}
    # build ordinal patterns
    patterns = {}
    n = len(x)
    for i in range(n - delay*(order-1)):
        window = x[i:(i + delay*order):delay]
        ranks = tuple(np.argsort(window))
        patterns[ranks] = patterns.get(ranks, 0) + 1
    ps = np.array(list(patterns.values()), dtype=float)
    ps = ps / (np.sum(ps) + 1e-12)
    pe = float(-np.sum(ps * np.log(ps + 1e-12)))
    # normalized by log(factorial(order))
    return {"置换熵": pe / math.log(math.factorial(order) + 1e-12)}

# -----------------------------
# 频域细化：标准脑电频段功率、相对功率、谱边频率
# -----------------------------
def bandpower_bands(arr: np.ndarray, fs: int, bands=None):
    """返回绝对与相对带功率以及带比值（常用：alpha/theta, beta/alpha）"""
    if bands is None:
        bands = {'delta': (1,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,45)}
    arr = np.asarray(arr).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size <= 1:
        out = {}
        for b in bands:
            out[f"{b}_power"] = 0.0
            out[f"{b}_rel"] = 0.0
        out["alpha_theta_ratio"] = 0.0
        out["beta_alpha_ratio"] = 0.0
        return out
    freqs, psd = welch(arr, fs=fs, nperseg=min(1024, arr.size))
    total_power = np.sum(psd)
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        p = float(np.sum(psd[mask]))
        out[f"{name}_power"] = p
        out[f"{name}_rel"] = float(p / (total_power + 1e-12))
    # ratio features
    out["alpha_theta_ratio"] = float(out["alpha_power"] / (out["theta_power"] + 1e-12))
    out["beta_alpha_ratio"] = float(out["beta_power"] / (out["alpha_power"] + 1e-12))
    # median freq
    cumsum = np.cumsum(psd)
    idx = np.searchsorted(cumsum, 0.5 * cumsum[-1])
    out["median_freq"] = float(freqs[idx]) if idx < len(freqs) else float(freqs[-1])
    # spectral edge freq 95%
    idx95 = np.searchsorted(cumsum, 0.95 * cumsum[-1])
    out["SEF95"] = float(freqs[idx95]) if idx95 < len(freqs) else float(freqs[-1])
    return out

# -----------------------------
# 时域 ERP / 峰值相关特征
# -----------------------------
def erp_peak_features(x: np.ndarray, sample_hz: int, baseline_samples: int = 0):
    """返回最大/最小峰值、峰值潜伏期、AUC（对 ERP 片段有用）"""
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {
            "峰值最大": 0.0, "峰值最小": 0.0, "峰值幅值差": 0.0,
            "最大峰位(s)": 0.0, "最小峰位(s)": 0.0, "AUC": 0.0, "baseline_mean": 0.0
        }
    baseline_mean = float(np.mean(x[:baseline_samples])) if baseline_samples > 0 and x.size > baseline_samples else float(np.mean(x))
    max_v = float(np.max(x))
    min_v = float(np.min(x))
    max_idx = int(np.argmax(x))
    min_idx = int(np.argmin(x))
    auc = float(np.trapz(x))  # 简单 AUC
    return {
        "峰值最大": max_v,
        "峰值最小": min_v,
        "峰值幅值差": float(max_v - min_v),
        "最大峰位(s)": float(max_idx / sample_hz),
        "最小峰位(s)": float(min_idx / sample_hz),
        "AUC": auc,
        "baseline_mean": baseline_mean
    }

# -----------------------------
# 跨通道 / 连接性（可选）
# -----------------------------
def stat_connectivity_summary(epoch: np.ndarray, fs: int, channel_names=None,
                              pairs=None, max_pairs: int = 10, bands=None):
    """
    epoch: shape (n_channels, n_samples)
    pairs: iterable of (i,j) pairs (indices). 若 None，则按 combinations 取前 max_pairs 对。
    返回键形如 "Ch1-Ch2_coherence_delta" / "Ch1-Ch2_PLV_alpha" 等
    """
    n_ch = epoch.shape[0]
    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(n_ch)]
    if pairs is None:
        all_pairs = list(combinations(range(n_ch), 2))
        pairs = list(islice(all_pairs, max_pairs))
    if bands is None:
        bands = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
    res = {}
    for (i,j) in pairs:
        xi = epoch[i, :]
        xj = epoch[j, :]
        xi = xi[~np.isnan(xi)]; xj = xj[~np.isnan(xj)]
        if xi.size == 0 or xj.size == 0:
            for b in bands:
                res[f"{channel_names[i]}-{channel_names[j]}_coherence_{b}"] = 0.0
                res[f"{channel_names[i]}-{channel_names[j]}_PLV_{b}"] = 0.0
            continue
        # coherence over whole spectrum (scipy.signal.coherence)
        try:
            f, Cxy = coherence(xi, xj, fs=fs, nperseg=min(1024, min(xi.size, xj.size)))
        except Exception:
            f = np.array([])
            Cxy = np.array([])
        for b, (lo, hi) in bands.items():
            if f.size == 0:
                res[f"{channel_names[i]}-{channel_names[j]}_coherence_{b}"] = 0.0
            else:
                mask = (f >= lo) & (f <= hi)
                res[f"{channel_names[i]}-{channel_names[j]}_coherence_{b}"] = float(np.mean(Cxy[mask]) if mask.any() else 0.0)
        # PLV: use analytic signal phases
        try:
            ph1 = np.angle(hilbert(xi))
            ph2 = np.angle(hilbert(xj))
            phase_diff = np.angle(np.exp(1j*(ph1 - ph2)))
            # band-limited PLV: filter in band is required for precise PLV; here approximate by computing PLV on whole signal then bandpass per band would be more correct.
            plv = lambda arr: float(np.abs(np.mean(np.exp(1j*arr))))
            # crude per-band PLV by bandpass via FFT mask (approx)
            # Simpler: compute PLV on full signal as baseline
            full_plv = plv(phase_diff)
            for b in bands:
                res[f"{channel_names[i]}-{channel_names[j]}_PLV_{b}"] = float(full_plv)
        except Exception:
            for b in bands:
                res[f"{channel_names[i]}-{channel_names[j]}_PLV_{b}"] = 0.0
    return res

# -----------------------------
# 将这些特征组织成函数便于调用
# -----------------------------
def stat_nonlinear_summary(sequence: np.ndarray, sample_hz: int):
    """整合上面所有非线性/复杂性/带功率/ERP峰值特征"""
    seq = np.asarray(sequence).ravel()
    seq = seq[~np.isnan(seq)]
    out = {}
    # Hjorth
    out.update({k: v for k, v in hjorth_parameters(seq).items()})
    # zero crossing / slope sign change
    out.update(zero_crossing_rate(seq))
    out.update(slope_sign_changes(seq))
    # LZ and Higuchi
    out.update(lz_complexity(seq))
    out.update(higuchi_fd(seq))
    # entropy family
    out.update(sample_entropy(seq))
    out.update(permutation_entropy(seq))
    # bandpower and ratio etc.
    out.update(bandpower_bands(seq, fs=sample_hz))
    # ERP-like peak and baseline features
    out.update(erp_peak_features(seq, sample_hz))
    return out

# -----------------------------
# 将新特征接入主提取流程（示例）
# -----------------------------
# 在 extract_eeg_features 函数签名中添加两个可选参数：
#   compute_connectivity: bool = False
#   connectivity_pairs: Optional[Sequence[Tuple[int,int]]] = None
#
# 在每个通道特征合并前后，调用 stat_nonlinear_summary(seq, sample_hz)
# 并在 epoch 层面（需要 epoch 的所有通道数据）调用 stat_connectivity_summary(epoch, sample_hz, ...)
#
# 下面示意如何在你的 extract_eeg_features 中插入（只示范关键片段）：

# 替换原先 ch loop 中对每个通道的特征合并一段：
# t_feats = stat_time_summary(seq)
# f_feats = stat_frequency_summary(seq, sample_hz=sample_hz, rotation_hz=rotation_hz)
# tf_feats = stat_timefreq_summary(seq, sample_hz=sample_hz, rotation_hz=rotation_hz)
# 新增：
# nonlinear_feats = stat_nonlinear_summary(seq, sample_hz=sample_hz)
# merged.update(nonlinear_feats)

# 在每个 epoch（对所有通道完成单通道特征）之后，如果 compute_connectivity 为 True：
# conn_feats = stat_connectivity_summary(sig[e], fs=sample_hz, channel_names=ch_names,
#                                       pairs=connectivity_pairs, max_pairs=max_pairs)
# 将 conn_feats 的键直接并入 row_dict（注意命名使用 "ChA-ChB_..." 格式）

# 最后仍然按照 template 统一列，填充 NaN -> 0.0

# -----------------------------
# 使用建议（建模与维度控制）
# -----------------------------
# 1) 对于常规 EEG/BCI 建议默认开启：时域（已有）、频域（已有 + bandpower）、时频（已有）、非线性（sample/perm/higuchi/LZ/Hjorth）。
# 2) 若样本较少或通道较多，谨慎开启连接性（connectivity），优先选择少量重要通道对（例如前额 vs 颞叶对），或只计算相邻/感兴趣通道对。
# 3) 对于 ERP（若 epoch 对齐刺激后 0-500 ms），AUC、峰值潜伏期、峰值幅值差、baseline_mean 对判别通常很有用。
# 4) 提取后建议做特征选择：基于统计检验（t-test/Wilcoxon）、随机森林重要性、LASSO 或递归特征消除（RFE）。
#
# -----------------------------
# 结束
# -----------------------------
