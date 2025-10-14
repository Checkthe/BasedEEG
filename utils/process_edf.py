import numpy as np
import pyedflib
import mne
from typing import Tuple, List, Optional
import warnings

def read_edf(psg_file_path: str,
                         hypnogram_file_path: str,
                         target_channels: Optional[List[str]] = None,
                         epoch_length: int = 30,
                         target_freq: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取Sleep-EDFx数据集的单个文件并处理成指定格式

    Parameters:
    -----------
    psg_file_path : str
        PSG数据文件路径 (*.edf)
    hypnogram_file_path : str
        睡眠分期标签文件路径 (*Hypnogram.edf)
    target_channels : List[str], optional
        目标通道列表，如果为None则使用默认通道
    epoch_length : int, default=30
        每个epoch的长度（秒）
    target_freq : int, default=100
        目标采样频率（Hz）

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        data: shape为(n_epochs, n_channels, n_points)的数据
        labels: shape为(n_epochs,)的标签数组
    """

    # 默认通道（常用的睡眠分析通道）
    if target_channels is None:
        #'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'
        target_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']

    # 读取PSG数据
    print(f"正在读取PSG文件: {psg_file_path}")
    with pyedflib.EdfReader(psg_file_path) as psg_file:
        # 获取通道信息
        channel_labels = psg_file.getSignalLabels()
        n_channels_total = psg_file.signals_in_file
        sample_frequencies = psg_file.getSampleFrequencies()

        # 找到目标通道的索引
        target_channel_indices = []
        actual_channels = []
        for target_ch in target_channels:
            found = False
            for i, ch_label in enumerate(channel_labels):
                if target_ch in ch_label or ch_label in target_ch:
                    target_channel_indices.append(i)
                    actual_channels.append(ch_label)
                    found = True
                    break
            if not found:
                print(f"警告: 未找到通道 {target_ch}")

        if not target_channel_indices:
            raise ValueError("未找到任何目标通道")

        print(f"使用通道: {actual_channels}")

        # 读取数据
        signals = []
        original_freqs = []
        for idx in target_channel_indices:
            signal = psg_file.readSignal(idx)
            signals.append(signal)
            original_freqs.append(sample_frequencies[idx])

        # 获取记录时长
        record_duration = psg_file.file_duration

    # 读取睡眠分期标签
    with pyedflib.EdfReader(hypnogram_file_path) as hyp_file:
        # 读取标注信息
        annotations = hyp_file.readAnnotations()
        sleep_stages = []

        # 处理标注
        for i in range(len(annotations[0])):  # annotations[0]是时间点，annotations[2]是标签
            onset_time = annotations[0][i]
            duration = annotations[1][i] if len(annotations[1]) > i else epoch_length
            stage_label = annotations[2][i] if len(annotations[2]) > i else 'Sleep stage W'

            # 将睡眠分期标签转换为数字
            if 'Sleep stage W' in stage_label:
                stage_num = 0  # 觉醒
            elif 'Sleep stage 1' in stage_label:
                stage_num = 1  # N1
            elif 'Sleep stage 2' in stage_label:
                stage_num = 2  # N2
            elif 'Sleep stage 3' in stage_label:
                stage_num = 3  # N3
            elif 'Sleep stage 4' in stage_label:
                stage_num = 3  # N4合并到N3
            elif 'Sleep stage R' in stage_label:
                stage_num = 4  # REM
            else:
                stage_num = -1  # 未知或运动伪迹

            sleep_stages.append((onset_time, duration, stage_num))

    # 重采样到目标频率
    resampled_signals = []
    for i, signal in enumerate(signals):
        original_freq = original_freqs[i]
        if original_freq != target_freq:
            # 简单的重采样（线性插值）
            original_length = len(signal)
            target_length = int(original_length * target_freq / original_freq)
            resampled_signal = np.interp(
                np.linspace(0, original_length - 1, target_length),
                np.arange(original_length),
                signal
            )
            resampled_signals.append(resampled_signal)
        else:
            resampled_signals.append(signal)

    # 计算每个epoch的采样点数
    n_points_per_epoch = epoch_length * target_freq

    # 根据睡眠分期标签分段数据
    n_epochs = len(sleep_stages)
    n_channels = len(resampled_signals)

    # 初始化输出数组
    data = np.zeros((n_epochs, n_channels, n_points_per_epoch))
    labels = np.zeros(n_epochs, dtype=int)

    valid_epochs = 0
    for epoch_idx, (onset_time, duration, stage_num) in enumerate(sleep_stages):
        # 计算在信号中的起始和结束位置
        start_sample = int(onset_time * target_freq)
        end_sample = int((onset_time + duration) * target_freq)

        # 检查是否超出信号范围
        if end_sample > len(resampled_signals[0]):
            continue

        # 提取数据段
        epoch_data = []
        for ch_idx in range(n_channels):
            signal_segment = resampled_signals[ch_idx][start_sample:end_sample]

            # 如果长度不足，进行填充或截断
            if len(signal_segment) < n_points_per_epoch:
                # 零填充
                padded_segment = np.zeros(n_points_per_epoch)
                padded_segment[:len(signal_segment)] = signal_segment
                signal_segment = padded_segment
            elif len(signal_segment) > n_points_per_epoch:
                # 截断
                signal_segment = signal_segment[:n_points_per_epoch]

            epoch_data.append(signal_segment)

        # 存储数据
        if valid_epochs < n_epochs:
            data[valid_epochs] = np.array(epoch_data)
            labels[valid_epochs] = stage_num
            valid_epochs += 1

    # 裁剪到有效的epochs
    data = data[:valid_epochs]
    labels = labels[:valid_epochs]

    # 过滤掉无效标签
    valid_mask = labels >= 0
    data = data[valid_mask]
    labels = labels[valid_mask]

    return data, labels


def load_p300_edf(file_path, tmin=0.0, tmax=0.8, target_codes=[9,13], non_target_codes=[4,5,6,7,8,10,11,12,14,15]):
    raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto', verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    codes = list(event_dict.values())

    if target_codes is None:
        target_codes = [max(codes)]
    if non_target_codes is None:
        non_target_codes = [c for c in codes if c not in target_codes]

    # 使用 list 形式
    all_codes = list(set(target_codes + non_target_codes))
    epochs = mne.Epochs(raw, events, event_id=all_codes,
                        tmin=0.0, tmax=0.8,
                        baseline=(0, 0),  # 或者 None
                        preload=True, verbose=False)

    data = epochs.get_data()
    codes_all = epochs.events[:, 2]
    # 二分类标签
    labels = np.array([1 if c in target_codes else 0 for c in codes_all])

    return data, labels

