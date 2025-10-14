import numpy as np
from scipy import signal


def compute_band_power(data, fs=250, freq_bands=None):
    """
    计算EEG信号的频带功率

    参数:
    data: numpy数组，形状为 (n_trials, n_channels, n_samples) 或 (n_channels, n_samples)
    fs: float, 采样频率 (Hz)
    freq_bands: dict, 频带定义，格式为 {'band_name': (low_freq, high_freq)}
                如果为None，使用默认的EEG频带

    返回:
    band_powers: numpy数组，各频带的功率值
                形状为 (n_trials, n_channels, n_bands) 或 (n_channels, n_bands)
    band_names: list, 频带名称列表
    """

    # 默认EEG频带定义
    if freq_bands is None:
        freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

    # 处理数据维度
    original_shape = data.shape
    if len(data.shape) == 2:
        # 单个试次: (n_channels, n_samples)
        data = data[np.newaxis, :, :]
        single_trial = True
    else:
        single_trial = False

    n_trials, n_channels, n_samples = data.shape
    n_bands = len(freq_bands)

    # 初始化结果数组
    band_powers = np.zeros((n_trials, n_channels, n_bands))
    band_names = list(freq_bands.keys())

    # 计算每个试次、每个通道、每个频带的功率
    for trial in range(n_trials):
        for ch in range(n_channels):
            # 获取当前通道的信号
            signal_data = data[trial, ch, :]

            # 计算功率谱密度
            freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(1024, n_samples // 4))

            # 计算各频带功率
            for i, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
                # 找到频带范围内的频率索引
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

                # 计算频带功率（积分功率谱密度）
                band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                band_powers[trial, ch, i] = band_power

    # 如果输入是单个试次，返回相应维度
    if single_trial:
        band_powers = band_powers[0]  # 形状变为 (n_channels, n_bands)
    return band_powers

def compute_relative_band_power(data, fs, freq_bands=None):
    """
    计算EEG信号的相对频带功率（各频带功率占总功率的比例）

    参数:
    data: numpy数组，形状为 (n_trials, n_channels, n_samples) 或 (n_channels, n_samples)
    fs: float, 采样频率 (Hz)
    freq_bands: dict, 频带定义

    返回:
    relative_powers: numpy数组，相对频带功率 (0-1之间)
    band_names: list, 频带名称列表
    """

    # 计算绝对功率
    band_powers, band_names = compute_band_power(data, fs, freq_bands)

    # 计算总功率（所有频带功率之和）
    total_power = np.sum(band_powers, axis=-1, keepdims=True)

    # 计算相对功率
    relative_powers = band_powers / (total_power + 1e-10)  # 添加小常数避免除零错误

    return relative_powers, band_names

