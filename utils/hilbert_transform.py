import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy.signal import butter, sosfiltfilt, hilbert
import warnings

def bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    """
    返回 bandpass 的 second-order-sections (sos)
    """
    nyq = fs / 2.0
    if low <= 0:
        raise ValueError("low must be > 0")
    if high >= nyq:
        raise ValueError(f"high must be < Nyquist ({nyq} Hz)")
    return butter(order, [low / nyq, high / nyq], btype='bandpass', output='sos')

def hilbert_band_power(
    eeg: np.ndarray,
    sampling_rate: float = 250.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    filter_order: int = 4,
    return_envelope: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    计算 EEG 在若干频段下的瞬时功率（Hilbert 变换法）。

    参数:
    ----------
    eeg : ndarray
        原始 EEG 数据。形状可以是:
         - (B, C, T) : B trials, C channels, T samples
         - (C, T)    : 等同于 B=1
         - (T,)      : 等同于 B=1, C=1
    sampling_rate : float
        采样率（Hz），默认 250.0
    bands : dict or None
        频段字典，键为名称，值为 (low, high) Hz。若为 None，使用默认:
            {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta' : (13, 30),
                'gamma': (30, 50)
            }
    filter_order : int
        Butterworth 滤波器阶数（每个段），默认 4
    return_envelope : bool
        如果 True，返回包络（幅值）；否则返回功率（幅值平方）。默认 False（返回功率）。

    返回:
    ----------
    band_result : ndarray
        形状 (B, C, n_bands, T) 的数组，类型 float64。
        若输入原为 (C,T) 或 (T,), 仍以 B 为 1 返回。
    band_names : list[str]
        频段名称列表，顺序对应 band_result 的第三维。
    """
    # 默认频段
    if bands is None:
        bands = {
            'delta': (1.0, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta' : (13.0, 30.0),
            'gamma': (30.0, 50.0)
        }

    # 规范化输入形状到 (B, C, T)
    eeg = np.asarray(eeg)
    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, np.newaxis, :]  # (1,1,T)
    elif eeg.ndim == 2:
        # could be (C,T) or (B,T) ambiguous; assume (C,T) => add batch dim
        eeg = eeg[np.newaxis, ...]  # (1,C,T)
    elif eeg.ndim == 3:
        pass
    else:
        raise ValueError("eeg must be 1D, 2D or 3D array")

    B, C, T = eeg.shape
    n_bands = len(bands)
    band_names = list(bands.keys())

    # 结果初始化
    band_result = np.zeros((B, C, n_bands, T), dtype=np.float64)

    # 检查 Nyquist
    nyq = sampling_rate / 2.0

    for idx, (name, (low, high)) in enumerate(bands.items()):
        # 验证频带有效性
        if low >= high:
            raise ValueError(f"band {name}: low must be < high")
        if low <= 0:
            warnings.warn(f"band {name}: low <= 0, setting to 0.01 Hz")
            low = 0.01
        if high >= nyq:
            warnings.warn(f"band {name}: high >= Nyquist ({nyq} Hz). Clipping to {nyq - 1e-6}")
            high = nyq - 1e-6
        if low >= high:
            warnings.warn(f"band {name} degenerate after clipping; result will be zeros")
            continue

        # 设计滤波器 sos
        try:
            sos = bandpass_sos(low, high, sampling_rate, order=filter_order)
        except ValueError as e:
            warnings.warn(f"无法设计 {name} 带通滤波器: {e}")
            continue

        # 对每个 trial*channel 应用滤波 + Hilbert
        # 为了效率，将 trial*channel 拉平成一维列表
        stacked = eeg.reshape(B * C, T)
        filtered = np.empty_like(stacked)
        for r in range(stacked.shape[0]):
            # zero-phase filtering
            filtered[r] = sosfiltfilt(sos, stacked[r])

        # Hilbert: 能对整个数组按最后一维并行处理
        analytic = hilbert(filtered, axis=-1)  # shape (B*C, T), complex
        if return_envelope:
            measure = np.abs(analytic)  # 幅值包络
        else:
            measure = np.abs(analytic) ** 2  # 瞬时功率

        # 恢复 (B, C, T) 并填入结果
        measure = measure.reshape(B, C, T)
        band_result[:, :, idx, :] = measure

    return band_result, band_names
