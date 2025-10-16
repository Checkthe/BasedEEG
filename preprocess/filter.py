import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def notch_filter(data, notch_freq, fs, quality_factor=30):
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, data, axis=-1)

from scipy.signal import butter, filtfilt

def butter_bandstop(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # btype='bandstop' 定义带阻滤波
    return butter(order, [low, high], btype='bandstop')

def butter_bandstop_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)  # 零相位处理，避免相位失真
    return y

def filtering(eeg_data, sfreq=250, notch_freq=50.0, bp_range=(8,30), bs_range=None):
    """
    对 EEG 信号进行滤波处理，滤除工频、眼电、肌电干扰

    参数:
        eeg_data: numpy.ndarray, 形状为 (n_trials, n_channels, n_samples)
        sfreq: float, 采样频率（Hz）
        notch_freq: float, 工频频率（默认为50Hz，可改为60Hz）

    返回:
        numpy.ndarray, 滤波后的 EEG 数据
    """
    n_trials, n_channels, n_samples = eeg_data.shape
    filtered_data = np.zeros_like(eeg_data)

    for trial in range(n_trials):
        for ch in range(n_channels):
            signal = eeg_data[trial, ch, :]
            signal = notch_filter(signal, notch_freq, sfreq)        # 去除工频
            signal = bandpass_filter(signal, bp_range[0], bp_range[1], sfreq)       # 带通滤波
            if bs_range:
                signal = butter_bandstop_filter(signal, bs_range[0], bs_range[1], fs=sfreq,order=4)  #带阻滤波
            filtered_data[trial, ch, :] = signal

    return filtered_data
