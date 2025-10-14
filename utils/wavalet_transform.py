import numpy as np
import pywt


def get_center_frequency(wavelet):
    try:
        if wavelet.startswith('cmor'):
            return float(wavelet.split('-')[1]) if '-' in wavelet else 1.0
        return pywt.central_frequency(wavelet)
    except Exception:
        return 1.0


def eeg_wavelet_transform(eeg_data, wavelet='cmor1.5-1.0', scales=None, sampling_rate=250):
    """
    批量小波变换（支持单 trial 或多 trial）

    参数:
    eeg_data: ndarray，形状为 (n_trials, n_channels, n_samples) 或 (n_channels, n_samples)
    wavelet: str，小波基函数（默认复 Morlet）
    scales: ndarray 或 None，自定义尺度；None 时自动覆盖 1-50Hz
    sampling_rate: float，采样率

    返回:
    result_dict: dict，包含以下键：
        'coefficients': ndarray, 小波系数 (n_trials, n_channels, n_scales, n_samples)
        'frequencies': ndarray, 对应频率 (n_scales,)
        'scales': ndarray, 使用的尺度
    """
    eeg_data = np.asarray(eeg_data)

    if eeg_data.ndim == 2:
        eeg_data = eeg_data[np.newaxis, ...]  # 增加 trial 维

    n_trials, n_channels, n_samples = eeg_data.shape

    if scales is None:
        freq_range = np.logspace(np.log10(1), np.log10(50), num=30)  # 1~50Hz, log-scale
        center_freq = get_center_frequency(wavelet)
        scales = center_freq * sampling_rate / freq_range
        scales = np.clip(scales, 1.0, 1000.0)
        scales = np.sort(scales)[::-1]
    else:
        scales = np.asarray(scales)
        if np.any(scales <= 0):
            raise ValueError("所有尺度值必须大于0")

    frequencies = get_center_frequency(wavelet) * sampling_rate / scales
    n_scales = len(scales)

    coefficients = np.zeros((n_trials, n_channels, n_scales, n_samples), dtype=np.complex128)

    for trial in range(n_trials):
        for ch in range(n_channels):
            try:
                coef, _ = pywt.cwt(eeg_data[trial, ch], scales, wavelet, sampling_period=1.0 / sampling_rate)
            except Exception:
                coef, _ = pywt.cwt(eeg_data[trial, ch], scales, wavelet)
            coefficients[trial, ch] = coef

    return {
        'coefficients': coefficients,
        'frequencies': frequencies,
        'scales': scales
    }


def compute_wavelet_power(coefficients):
    """
    小波功率谱 (取模平方)

    参数:
    coefficients: ndarray, 形状为 (n_trials, n_channels, n_scales, n_samples)

    返回:
    ndarray, 同形状功率谱
    """
    return np.abs(coefficients) ** 2


def extract_frequency_bands(coefficients, frequencies, reduce_time=False):
    """
    提取典型 EEG 频段功率谱 (支持平均或保留时间分辨率)

    参数:
    coefficients: ndarray, 小波系数 (n_trials, n_channels, n_scales, n_samples)
    frequencies: ndarray, 频率数组 (n_scales,)
    reduce_time: bool，是否对时间轴做均值

    返回:
    band_power_matrix: ndarray，频段功率
        - reduce_time=False: (n_trials, n_channels, n_bands, n_samples)
        - reduce_time=True:  (n_trials, n_channels, n_bands)
    band_names: list[str]，频段标签
    """
    power = compute_wavelet_power(coefficients)

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 50)
    }

    band_names = []
    band_powers_list = []

    for band, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies <= high)
        if not np.any(mask):
            print(f"警告: 频段 {band} ({low}-{high} Hz) 超出频率范围")
            shape = (power.shape[0], power.shape[1], power.shape[3]) if not reduce_time else (power.shape[0], power.shape[1])
            band_power = np.zeros(shape)
        else:
            band_power = np.mean(power[:, :, mask, :], axis=2)  # (n_trials, n_channels, n_samples)
            if reduce_time:
                band_power = np.mean(band_power, axis=2)  # (n_trials, n_channels)

        band_names.append(band)
        band_powers_list.append(band_power)

    band_power_matrix = np.stack(band_powers_list, axis=2 if reduce_time else 2)

    return band_power_matrix, band_names


def wavelet_transform(eeg_data, sampling_rate=250, wavelet='cmor1.5-1.0', reduce_time=False):
    """
    脑电小波分析主函数 (支持批处理 trial)

    参数:
    eeg_data: ndarray, 形状为 (n_trials, n_channels, n_points) 或 (n_channels, n_points)
    sampling_rate: float, 采样率
    wavelet: str, 小波名称
    reduce_time: bool, 是否对时间维做平均

    返回:
    result: dict, 包含小波系数、频率、尺度、频段功率
    """
    result = eeg_wavelet_transform(eeg_data, wavelet=wavelet, sampling_rate=sampling_rate)
    band_powers, band_names = extract_frequency_bands(result['coefficients'], result['frequencies'], reduce_time=reduce_time)
    result['band_powers'] = band_powers
    result['band_names'] = band_names
    return result
