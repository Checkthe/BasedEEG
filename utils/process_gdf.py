import mne
import numpy as np
from typing import Tuple, Optional, Union, Dict
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def read_gdf_2b(
    file_path: str,
    t_min: float = 0.0,
    t_max: float = 4.0,
    if_drop_eog: bool = False,
    if_labels: bool = True,
    explicit_event_map: Optional[Dict[str, Union[str, int]]] = None,
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    resample_sfreq: Optional[float] = None,
    picks_channels: Optional[list] = None,
    baseline: Optional[Tuple[float, float]] = None,
    verbose: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    读取 BCIC IV 2b 风格的 GDF 文件并返回 epochs 数据（和可选标签）。

    返回:
      (data, labels) 或 data
      data shape: (n_epochs, n_channels, n_times)
      labels: 0/1 integers where 0=left, 1=right

    参数说明:
      file_path: gdf 文件路径
      t_min, t_max: epoch 时间窗口（相对于事件 onset，单位秒）。例如 t_min=1.25, t_max=5.25。
      if_drop_eog: 是否删除 EOG 通道（根据常见名称启发式判断）
      if_labels: 是否返回标签（若为 False 则只返回 data）
      explicit_event_map: 可选显式映射，支持三种形式:
          - {'left': 4, 'right': 5}  （value 为 event code int）
          - {'T1': 4, 'T2': 5}      （annotation name -> code）
          - {'left': 'T1', 'right': 'T2'} （提供 annotation 名称）
      l_freq, h_freq: optional bandpass limits passed to mne.Raw.preprocess
      resample_sfreq: optional new sampling rate (Hz)
      picks_channels: optional list of channel names to use (覆盖自动选择)
      baseline: optional baseline window (tmin, tmax) relative to epoch for baseline correction
      verbose: 是否将 mne 的 verbose 打开

    注意:
      - 当您对数据做重采样时，务必保证在重采样前有合适的低通以避免混叠。
      - 函数会尝试智能识别 left/right 注释（常见 'T1','T2' 或 '4','5'），若无法识别会抛出异常并显示当前 annotation->code 映射。
    """
    # 1) 读取 raw
    try:
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=verbose)
    except Exception as e:
        raise RuntimeError(f"读取文件 {file_path} 时出错: {e}")

    # 2) 可选删除 EOG 通道（启发式匹配）
    if if_drop_eog:
        eog_patterns = ['eog', 'heog', 'veog', 'eye', 'left eog', 'right eog']
        drop_chs = [ch for ch in raw.ch_names if any(p in ch.lower() for p in eog_patterns)]
        if drop_chs:
            raw.drop_channels(drop_chs)
            if verbose:
                print(f"Dropped EOG channels: {drop_chs}")

    # 3) 滤波（若用户提供）
    if l_freq is not None or h_freq is not None:
        # mne.preprocess: use keyword args for clarity
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=verbose)

    # 4) 重采样（若用户提供）
    if resample_sfreq is not None:
        # 推荐在调用此函数前就已保证适当低通；这里直接调用 MNE 的 resample
        raw.resample(sfreq=resample_sfreq, npad='auto', verbose=verbose)

    # 5) annotations -> events 映射
    events, event_id_mapping = mne.events_from_annotations(raw, verbose=verbose)
    if verbose:
        print("Annotation->code mapping:", event_id_mapping)

    # 6) 决定 left/right 的 annotation keys
    left_key = right_key = None

    # 如果 explicit_event_map 提供
    if explicit_event_map is not None:
        inv = {v: k for k, v in event_id_mapping.items()}
        # case A: user 给出 numeric codes for left/right e.g. {'left':4,'right':5}
        if all(isinstance(v, int) for v in explicit_event_map.values()):
            left_val = explicit_event_map.get('left') or explicit_event_map.get('Left')
            right_val = explicit_event_map.get('right') or explicit_event_map.get('Right')
            # allow form {'T1':4,'T2':5} too: try to map keys that equal existing annotation names
            if left_val in inv:
                left_key = inv[left_val]
            if right_val in inv:
                right_key = inv[right_val]
            # also support explicit_event_map giving {'T1':4,'T2':5}: if any key matches annotation names, use them
            for k in explicit_event_map.keys():
                if k in event_id_mapping:
                    kl = k.lower()
                    if ('left' in kl or 'lhand' in kl or 't1' in kl) and left_key is None:
                        left_key = k
                    if ('right' in kl or 'rhand' in kl or 't2' in kl) and right_key is None:
                        right_key = k
        else:
            # case B: user provided names like {'left':'T1','right':'T2'} or {'T1':'T2'}
            for k, v in explicit_event_map.items():
                # if v is str and exists in mapping, assign
                if isinstance(v, str) and v in event_id_mapping:
                    if 'left' in k.lower() or 'left' in v.lower() or 't1' in v.lower():
                        left_key = v
                    if 'right' in k.lower() or 'right' in v.lower() or 't2' in v.lower():
                        right_key = v
                # direct mapping if key itself equals annotation
                if k in event_id_mapping:
                    kl = k.lower()
                    if 'left' in kl or 'lhand' in kl or 't1' in kl:
                        left_key = k
                    if 'right' in kl or 'rhand' in kl or 't2' in kl:
                        right_key = k

    # 启发式搜索 annotation key
    if left_key is None or right_key is None:
        for k in event_id_mapping.keys():
            kl = k.lower()
            if left_key is None and ('left' in kl or 'lhand' in kl or 'left hand' in kl or 't1' in kl or 'class1' in kl):
                left_key = k
            if right_key is None and ('right' in kl or 'rhand' in kl or 'right hand' in kl or 't2' in kl or 'class2' in kl):
                right_key = k

    # 作为最后的尝试，若 mapping 包含 code 4 与 5（BCI-IV 常见），映射回去
    if left_key is None or right_key is None:
        inv = {v: k for k, v in event_id_mapping.items()}
        if 4 in inv and 5 in inv:
            if left_key is None:
                left_key = inv[4]
            if right_key is None:
                right_key = inv[5]

    if left_key is None or right_key is None:
        raise ValueError(
            "无法自动识别 left/right 事件。现有 annotation->code 映射为: "
            f"{event_id_mapping}. 请通过 explicit_event_map 参数显式传入 mapping（例如 "
            "{'left':4,'right':5} 或 {'left':'T1','right':'T2'}）."
        )

    event_id = {left_key: event_id_mapping[left_key], right_key: event_id_mapping[right_key]}

    # 7) picks: 用户指定优先，否者自动挑选 EEG 通道（排除 EOG）
    if picks_channels is None:
        picks = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
        if len(picks) == 0:
            raise RuntimeError(f"未找到 EEG 通道（raw.ch_names={raw.ch_names}）。可使用 picks_channels 参数传入通道名列表。")
    else:
        # convert channel names to picks indices and validate
        missing = [ch for ch in picks_channels if ch not in raw.ch_names]
        if missing:
            raise RuntimeError(f"指定的 picks_channels 中包含 raw 中不存在的通道: {missing}")
        picks = [raw.ch_names.index(ch) for ch in picks_channels]

    # 8) 创建 epochs
    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=t_min,
            tmax=t_max,
            proj=False,
            picks=picks,
            baseline=baseline,
            preload=True,
            verbose=verbose
        )
    except Exception as e:
        raise RuntimeError(f"创建 Epochs 失败: {e}. 可用的 annotation->code 映射: {event_id_mapping}")

    data = epochs.get_data(copy=False)  # shape (n_epochs, n_channels, n_times)

    if if_labels:
        event_codes = epochs.events[:, -1]
        code_to_label = {
            event_id[left_key]: 0,
            event_id[right_key]: 1
        }
        # 如果 event_codes 中出现未知 code，立即报错
        labels = np.array([code_to_label.get(c, -1) for c in event_codes], dtype=np.int64)
        if (labels == -1).any():
            unknown = np.unique(event_codes[labels == -1])
            raise RuntimeError(f"存在未知 event code: {unknown}. 当前映射为: {event_id_mapping}")
        return data, labels
    else:
        return data


def read_gdf_2a(file_path: str, t_min: float = 0.0, t_max: float = 8.0, if_drop=True, if_labels=True) -> np.ndarray:
    try:
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return np.array([])
    # 安全删除 EOG 通道（如果存在）
    if if_drop:
        drop_chs = [ch for ch in ['EOG-left', 'EOG-central', 'EOG-right'] if ch in raw.ch_names]
        if drop_chs:
            raw.drop_channels(drop_chs)
    events, event_id_mapping = mne.events_from_annotations(raw, verbose=False)
    # event_ids = {'left_hand': 7, 'right_hand': 8, 'feet': 9, 'tongue': 10}
    print(event_id_mapping)
    print(events)
    event_ids = {'left_hand': 7, 'right_hand': 8, 'feet': 9, 'tongue': 10}

    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=t_min, tmax=t_max,
                        proj=False, picks='eeg', baseline=None, preload=True, verbose=False)
    data_in_epochs = epochs.get_data(copy=False)
    #获取标签
    if if_labels:
        labels = epochs.events[:, -1]
        unique_labels = np.unique(labels)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        mapped_labels = np.array([label_mapping[label] for label in labels])
        return data_in_epochs,mapped_labels
    else:
        return data_in_epochs