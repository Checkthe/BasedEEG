import numpy as np
from typing import Iterable, Optional, Sequence, Union

def apply_car(data: np.ndarray,
              exclude_channels: Optional[Sequence[int]] = None,
              weights: Optional[np.ndarray] = None,
              use_median: bool = False,
              in_place: bool = False) -> np.ndarray:
    """
    对 (B, C, N) 格式的 EEG 数据应用 Common Average Reference (CAR)。

    参数
    ----
    data : np.ndarray
        输入数据，shape = (B, C, N). dtype 应为 float32/float64 等数值类型。
    exclude_channels : sequence[int] or None, optional
        要从参考计算中排除的通道索引（例如坏道）。索引按 0..C-1。
        如果为 None，则使用所有通道。
    weights : np.ndarray or None, optional
        可选的权重数组用于计算加权平均参考。形状应为 (C,) 或 (B, C)。
        若提供，则 `exclude_channels` 将被应用（对应权重置 0）。
    use_median : bool, optional
        若为 True，则使用每时间点的通道中位数作为参考（更鲁棒，适用于含离群值的情况）。
        否则使用（加权）均值。
    in_place : bool, optional
        若为 True，则在原数组上进行修改并返回该数组；否则返回数据的拷贝（默认）。

    返回
    ----
    out : np.ndarray
        CAR 后的数据，shape = (B, C, N)。如果 in_place=False，返回的是新的数组副本。

    注意与实现细节
    ----------------
    - 参考是针对每个 batch/trial 单独计算的：对于每个 b，参考是对 data[b, :, :] 在通道维上按时间点计算的 (C->1) 序列。
    - 若存在 NaN 值，均值/中位数的计算会忽略 NaN（使用 nanmean / nanmedian）。
    - 若 C == 1，则函数会返回全零（通道减去自身均值）。
    - weights 支持按通道或按 batch-通道的权重（例如对不同 trial 采用不同权重），但不能对时间点加权。
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("`data` must be a numpy ndarray with shape (B, C, N).")
    if data.ndim != 3:
        raise ValueError("`data` must have 3 dimensions (B, C, N).")
    B, C, N = data.shape
    if C < 1:
        raise ValueError("Number of channels C must be >= 1.")
    if exclude_channels is None:
        exclude_mask = np.zeros(C, dtype=bool)
    else:
        exclude_mask = np.zeros(C, dtype=bool)
        for ch in exclude_channels:
            if ch < 0 or ch >= C:
                raise IndexError(f"exclude channel index {ch} out of range [0, {C-1}].")
            exclude_mask[ch] = True

    # Prepare output array
    out = data if in_place else data.copy()

    # Validate weights if provided
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim == 1:
            if weights.shape[0] != C:
                raise ValueError("weights shape (C,) expected when 1D.")
            weights_2d = np.tile(weights[None, :], (B, 1))  # (B, C)
        elif weights.ndim == 2:
            if weights.shape != (B, C):
                raise ValueError("weights shape must be (C,) or (B, C).")
            weights_2d = weights
        else:
            raise ValueError("weights must be 1D (C,) or 2D (B,C).")
        # 应用 exclude：将这些通道权重置 0
        weights_2d[:, exclude_mask] = 0.0
        # 若全部为 0（没有可用通道）则报错
        if np.allclose(weights_2d, 0):
            raise ValueError("All weights are zero after applying exclude_channels.")
    else:
        weights_2d = None

    # 主循环：按 batch 计算参考并减去（向量化）
    # 使用 nanmean / nanmedian 以兼容缺失值
    if use_median:
        # nanmedian 不支持 axis 对多维进行直接广播在早期 numpy 版本上表现不同，
        # 我们逐批次计算以保证兼容性和清晰性。
        for b in range(B):
            arr = out[b]  # shape (C, N)
            # 排除通道：将排除通道值设为 NaN，使 nanmedian 忽略它们
            if exclude_mask.any():
                arr_for_ref = arr.copy()
                arr_for_ref[exclude_mask, :] = np.nan
            else:
                arr_for_ref = arr
            # 计算每时间点的中位数参考，忽略 NaN
            ref = np.nanmedian(arr_for_ref, axis=0)  # shape (N,)
            # 若存在全部为 NaN 的时间点（理论上不应发生），用 0 填充
            ref = np.where(np.isnan(ref), 0.0, ref)
            out[b] = arr - ref[None, :]
    else:
        # 均值路径（支持加权）
        for b in range(B):
            arr = out[b]  # shape (C, N)
            if weights_2d is not None:
                w = weights_2d[b]  # shape (C,)
                # 将被排除的通道权重已经置 0
                # 为了在存在 NaN 的情况下稳定计算，我们对每时间点分别计算加权平均忽略 NaN：
                # 计算有效权重（把对应 NaN 的通道权重设为 0）
                nan_mask = np.isnan(arr)  # shape (C, N)
                w_masked = w[:, None] * (~nan_mask).astype(float)  # (C, N), 权重为0表示该通道在该时刻无效
                # product w * x，忽略 nan（将 nan 视为 0）
                arr_zeroed = np.where(nan_mask, 0.0, arr)
                numerator = (w_masked * arr_zeroed).sum(axis=0)   # (N,)
                denom = w_masked.sum(axis=0)                     # (N,)
                # 防除 0
                denom_safe = np.where(denom == 0.0, 1.0, denom)
                ref = numerator / denom_safe
                ref = np.where(denom == 0.0, 0.0, ref)  # 若 denom==0（无有效通道），将参考设为0
            else:
                # 普通均值，忽略 NaN
                if exclude_mask.any():
                    arr_for_ref = arr.copy()
                    arr_for_ref[exclude_mask, :] = np.nan
                else:
                    arr_for_ref = arr
                ref = np.nanmean(arr_for_ref, axis=0)  # (N,)
                ref = np.where(np.isnan(ref), 0.0, ref)
            out[b] = arr - ref[None, :]

    return out
