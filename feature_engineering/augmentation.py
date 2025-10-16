from typing import Union, Sequence, Any, Tuple
import numpy as np

from typing import Any, Sequence, Tuple, Union
from collections import defaultdict, Counter
import numpy as np

def augment_category_balance(
    data: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Sequence[Any],
    target_per_class: int = None,
    mix_ratio: float = 0.5,
    random_state: int = 42,
    add_small_noise: bool = False,
    noise_scale: float = 1e-6,
    allow_mixing_of_different_spatial_shapes: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对批量 time-series 数据按类别做“随机二值掩码混合”增广（只做上采样以平衡类别）。
    支持样本为 1D/2D/3D（时间轴为最后一维）。返回增广后的数据与标签。

    参数
    ----
    data : np.ndarray 或 Sequence[np.ndarray]
        批量样本，可为：
        - ndarray，形状 (N, ...)，函数会把每一项当作一个样本拆成 list；
        - 或者是序列（list/tuple/iterable）每项为 ndarray（或可转为 ndarray）。
        注意：如果输入为 1D ndarray (N,) 则视为 N 个标量样本。
    labels : Sequence[Any]
        与样本 1:1 对应的标签。
    target_per_class : int 或 None
        目标每类样本数。若为 None，则取原始样本中数量最多的类的样本数。
    mix_ratio : float in [0,1]
        掩码中来自第一个被采样样本 xa 的比例（沿时间轴）。
    random_state : int
        随机种子，用于可复现性。
    add_small_noise : bool
        是否在合成样本上添加微小高斯噪声。
    noise_scale : float
        噪声强度相对于信号标准差的比例（若信号 std 为 0，则以 1.0 为基准）。
    allow_mixing_of_different_spatial_shapes : bool
        是否允许在同类内混合形状不同（非时间轴）的样本。
        - False（默认）：若两样本在除时间轴之外的形状不同，则跳过该次合成并打印警告；
        - True：尝试依赖 numpy 广播进行合成（若广播失败也会跳过）。

    返回
    ----
    combined_data_nd : np.ndarray
        - 若所有样本形状一致 -> 数值型 ndarray，形状 (N_total, ...);
        - 否则 -> dtype=object 的 1D ndarray，每项为对应的 ndarray。
    combined_labels : np.ndarray
        dtype=object 的 1D ndarray，与 combined_data_nd 对应。
    """
    # 标准化输入为 list of ndarray
    if isinstance(data, np.ndarray) and data.ndim >= 2:
        data_list = [np.asarray(data[i]).copy() for i in range(data.shape[0])]
    else:
        data_list = [np.asarray(x).copy() for x in data]

    n = len(data_list)
    if n == 0:
        raise ValueError("输入 data 为空。")
    if len(labels) != n:
        raise ValueError("data 和 labels 长度不一致。")

    labels_arr = np.array(labels, dtype=object)

    # --- 打印处理前的类别数量（用户要求） ---
    try:
        pre_counter = Counter(labels_arr)
        print("处理前各类样本数量：")
        for c, num in sorted(pre_counter.items(), key=lambda x: str(x[0])):
            print(f"  {c}: {num}")
    except Exception:
        # 若计数失败则静默继续（避免中断主流程）
        pass

    # 每类索引池
    pool_idx = defaultdict(list)
    for idx, lab in enumerate(labels_arr):
        pool_idx[lab].append(idx)

    unique, counts = np.unique(labels_arr, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    if target_per_class is None:
        target_per_class_effective = int(np.max(counts)) if len(counts) > 0 else 0
    else:
        try:
            tp = int(target_per_class)
            target_per_class_effective = tp if tp > 0 else int(np.max(counts))
        except Exception:
            target_per_class_effective = int(np.max(counts))

    rng = np.random.default_rng(random_state)

    need_augment = {cls: max(0, target_per_class_effective - counts_dict.get(cls, 0)) for cls in unique}

    aug_data = []
    aug_labels = []

    for cls, n_needed in need_augment.items():
        if n_needed <= 0:
            continue
        pool = pool_idx.get(cls, [])
        if len(pool) == 0:
            print(f"警告：类别 {cls} 在原始数据中不存在，无法增广。")
            continue
        if len(pool) == 1:
            print(f"提示：类别 {cls} 只有 1 个原始样本，增广将以同一条信号自混合方式进行。")

        for _ in range(n_needed):
            ia = rng.choice(pool)
            ib = rng.choice(pool)

            xa = np.asarray(data_list[ia])
            xb = np.asarray(data_list[ib])

            # 至少为 1D（时间轴）；标量视为长度 1 序列
            if xa.ndim == 0:
                xa = xa.reshape((1,))
            if xb.ndim == 0:
                xb = xb.reshape((1,))

            L = min(xa.shape[-1], xb.shape[-1])
            if L == 0:
                print(f"警告：类别 {cls} 的候选样本存在空信号，跳过一次合成。")
                continue

            xa_ = xa[..., :L]
            xb_ = xb[..., :L]

            # 检查非时间轴（spatial）形状一致性
            spatial_xa = xa_.shape[:-1]
            spatial_xb = xb_.shape[:-1]
            if spatial_xa != spatial_xb:
                if not allow_mixing_of_different_spatial_shapes:
                    print(f"警告：类别 {cls} 的两个候选样本在非时间轴形状不同（{spatial_xa} vs {spatial_xb}），跳过该次合成。")
                    continue
                # else: 允许广播，继续尝试（若广播失败会被捕获）

            # 产生掩码并广播
            mask_1d = rng.random(L) < float(mix_ratio)
            mask_shape = (1,) * (xa_.ndim - 1) + (L,)
            mask = mask_1d.reshape(mask_shape)

            try:
                x_new = np.where(mask, xa_, xb_)
            except Exception as e:
                print(f"警告：合成时发生广播错误（类别 {cls}，索引 {ia},{ib}），跳过。错误信息: {e}")
                continue

            if add_small_noise:
                std_x = np.std(x_new)
                sigma = (std_x if std_x > 0 else 1.0) * float(noise_scale)
                noise = rng.normal(0.0, sigma, size=x_new.shape)
                x_new = x_new + noise

            aug_data.append(np.asarray(x_new))
            aug_labels.append(cls)

    combined_list = list(data_list) + aug_data
    combined_labels = np.concatenate([labels_arr, np.array(aug_labels, dtype=object)]) if len(aug_labels) > 0 else labels_arr

    # 尝试堆叠为数值型 ndarray
    shapes = [np.asarray(x).shape for x in combined_list]
    all_equal_shape = all(s == shapes[0] for s in shapes) if len(shapes) > 0 else True

    if all_equal_shape:
        try:
            combined_data_nd = np.stack(combined_list, axis=0)
        except Exception:
            combined_data_nd = np.array(combined_list, dtype=object)
    else:
        combined_data_nd = np.array(combined_list, dtype=object)

    # 输出增广后各类数量
    try:
        post_counter = Counter(combined_labels)
        print("均衡后各类数量：")
        for c, num in sorted(post_counter.items(), key=lambda x: str(x[0])):
            print(f"  {c}: {num}")
        print(f"(目标 target_per_class_effective = {target_per_class_effective})")
    except Exception:
        pass

    return combined_data_nd, combined_labels


if __name__ == "__main__":
    from config import cfg
    import pandas as pd
    cfgs = cfg().get_args()
    dataset = pd.read_csv(cfgs.features_path).iloc[:,1:]
    data = dataset.drop(columns=['labels'])
    labels = dataset['labels']
    feats = augment_category_balance(data=data.to_numpy(),labels=labels)
    feats = pd.DataFrame(feats,columns=data.columns)
    feats['labels'] = labels
    feats.to_csv(cfgs.features_path)