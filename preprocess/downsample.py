import numpy as np
from scipy.signal import firwin, filtfilt, resample_poly
from math import gcd


def fir_resample(data: np.ndarray,
                                orig_sfreq: float,
                                target_sfreq: int = 128,
                                l_freq: float = 1.0,
                                h_freq: float = 40.0,
                                numtaps: int | None = None,
                                verbose: bool = False,
                                do_fir: bool = True) -> np.ndarray:
    """
    对 (B, C, N) 格式的 EEG 批量数据进行可选的 1-40 Hz FIR 带通滤波并重采样到 target_sfreq。

    参数
    ----
    data : np.ndarray
        输入数据，形状 (B, C, N)。dtype 建议为 float32/float64。
    orig_sfreq : float
        原始采样率（Hz），必须提供。
    target_sfreq : int, optional
        目标采样率（Hz），默认 128。
    l_freq : float, optional
        带通下限频率（Hz），默认 1.0。仅在 do_fir=True 时生效。
    h_freq : float, optional
        带通上限频率（Hz），默认 40.0。仅在 do_fir=True 时生效。
    numtaps : int | None, optional
        FIR tap 数（滤波器长度）。若为 None，函数将根据 orig_sfreq 自动选取默认值。
        注意：若 numtaps 过大且序列较短，函数会自动缩短以避免 filtfilt 错误。仅在 do_fir=True 时生效。
    verbose : bool, optional
        若为 True，打印处理信息。
    do_fir : bool, optional
        是否执行 FIR 带通滤波（使用 firwin + filtfilt）。若为 False，将跳过滤波，只进行重采样（若需）。
        默认 True。

    返回
    ----
    out : np.ndarray
        处理后数据，形状 (B, C, new_N)，new_N 由重采样结果决定。

    实现细节与注意事项
    ------------------
    - 使用 firwin 设计带通 FIR（线性相位），再用 filtfilt 做 zero-phase 滤波以避免相位延迟（仅在 do_fir=True）。
    - 使用 resample_poly(up=target_sfreq, down=orig_sfreq) 做多相重采样（内部做 anti-aliasing）。
    - 若 orig_sfreq == target_sfreq 且 do_fir=True，则仅做带通滤波（不重采样）；若 do_fir=False，则直接返回原始数据（仅规范 dtype）。
    - 若数据很短而 numtaps 导致 filtfilt 报错，函数会自动减小 numtaps（但至少为 3）。
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a numpy array with shape (B, C, N).")
    if data.ndim != 3:
        raise ValueError("`data` must have shape (B, C, N).")
    B, C, N = data.shape
    if verbose:
        print(f"Input shape: B={B}, C={C}, N={N}, orig_sfreq={orig_sfreq} Hz, target_sfreq={target_sfreq} Hz, do_fir={do_fir}")
    if do_fir:
        if l_freq <= 0 or h_freq <= l_freq:
            raise ValueError("Invalid bandpass frequencies: require 0 < l_freq < h_freq.")
        nyq = orig_sfreq / 2.0
        if h_freq >= nyq:
            raise ValueError(f"h_freq ({h_freq} Hz) must be < Nyquist ({nyq} Hz) of orig_sfreq.")

        # 选择默认 numtaps（若未给定）：与采样率成比例的经验值
        if numtaps is None:
            # 经验选择：滤波器长度约为 0.3 * orig_sfreq（较宽）；随后会根据 N 自动缩短以保证 filtfilt 正常工作
            numtaps = max(3, int(round(0.3 * orig_sfreq)))
        # numtaps 必须为奇数以保证对称（线性相位）
        if numtaps % 2 == 0:
            numtaps += 1

        # 确保滤波器长度不会过大以致 filtfilt 报错（filtfilt 要求 padlen < N）
        # filtfilt 内部使用 padlen = 3*(max(len(a),len(b))-1)，这里 a=[1], b=numtaps taps
        max_padlen = max(1, N - 1)
        max_taps_allowed = max(3, (max_padlen // 3) + 1)  # 保守上限
        if numtaps > max_taps_allowed:
            if verbose:
                print(f"Warning: requested numtaps={numtaps} too large for sequence length N={N}; "
                      f"reducing to {max_taps_allowed} to allow filtfilt.")
            numtaps = max_taps_allowed
            if numtaps % 2 == 0:
                numtaps += 1

        # 设计带通 FIR（firwin 接受归一化频率: 0..1 对应 0..nyq）
        cutoff = [l_freq / nyq, h_freq / nyq]
        b = firwin(numtaps, cutoff, pass_zero=False)  # 带通

        if verbose:
            print(f"Using FIR numtaps={numtaps}, band={l_freq}-{h_freq} Hz")

        # 对批次数据做滤波（沿 samples 轴 axis=2）
        data_proc = data.astype(np.float64, copy=False)  # filtfilt 更稳定用 float64
        filtered = np.empty_like(data_proc)
        for i in range(B):
            x = data_proc[i]  # shape (C, N)
            try:
                filtered[i] = filtfilt(b, [1.0], x, axis=1)
            except Exception as e:
                if verbose:
                    print(
                        f"filtfilt failed for batch {i} with error: {e}. Falling back to causal lfilter (has phase delay).")
                from scipy.signal import lfilter
                filtered[i] = lfilter(b, [1.0], x, axis=1)
    else:
        # 不做 FIR，仅做重采样（或直接返回）
        if numtaps is not None and verbose:
            print("do_fir=False -> ignoring numtaps and band parameters.")
        # 快速路径：若 orig_sfreq == target_sfreq 且不需要 FIR，直接返回输入（但保证 dtype）
        if int(orig_sfreq) == int(target_sfreq):
            if verbose:
                print("do_fir=False and orig_sfreq == target_sfreq -> returning input without filtering/resampling.")
            return data.astype(data.dtype, copy=False)
        # 否则只做重采样（先把数据转换为 float64 以便后续处理）
        filtered = data.astype(np.float64, copy=False)
        if verbose:
            print("do_fir=False -> skipped filtering, proceeding to resampling only.")

    # 到这里，`filtered` 已经是要用于重采样的数组（dtype float64），形状 (B, C, N)
    # 若目标采样率与原始相同（在 do_fir=True 且经过滤波的情况），则仅返回滤波结果（并转回原 dtype）
    if do_fir and int(orig_sfreq) == int(target_sfreq):
        if verbose:
            print("orig_sfreq == target_sfreq, skipping resampling (returned filtered data).")
        return filtered.astype(data.dtype, copy=False)

    # 重采样：使用 resample_poly，up/down 为 target/orig 的整数比，先约分
    up = int(target_sfreq)
    down = int(orig_sfreq)
    g = gcd(up, down)
    up //= g
    down //= g
    if verbose:
        print(f"Resampling with up={up}, down={down} (reduced ratio after gcd).")

    # resample_poly 支持对多维数组指定 axis
    # 数据目前形状 (B, C, N) -> 我们希望对 axis=2 进行重采样
    # 为避免内存峰值，逐 batch 处理
    resampled_list = []
    for i in range(B):
        y = resample_poly(filtered[i], up=up, down=down, axis=1)
        resampled_list.append(y)
    out = np.stack(resampled_list, axis=0)

    if verbose:
        B2, C2, N2 = out.shape
        print(f"Output shape: (B={B2}, C={C2}, N={N2}), dtype={out.dtype}")

    # 返回与输入同 dtype（通常 float32/float64）
    return out.astype(data.dtype, copy=False)
