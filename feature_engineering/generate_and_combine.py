import torch
import numpy as np
import pandas as pd
from grpc.beta.implementations import insecure_channel
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from modeling.DownsamplingDAE import EEGDenoisingAutoencoder
from config import cfg
cfgs = cfg().get_args()
from typing import Union, Sequence, Any, List, Tuple

def augment_time_series(
    data: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Sequence[Any],
    target_per_class: int = None,  # 若为 None 则自动设为原始数据中样本最多的类的数量
    mix_ratio: float = 0.5,
    random_state: int = 42,
    add_small_noise: bool = False,
    noise_scale: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对批量 time-series 数据按类别做“随机二值掩码混合”增广。

    返回
    ----
    combined_data_nd : np.ndarray
        - 如果所有信号等长：返回形状 (N_total, L) 的 2D ndarray，dtype 为数值类型；
        - 如果信号长度不一：返回 1D ndarray，dtype=object，每项为对应的 1D ndarray（保持顺序）。
      原始样本在前，增广样本在后；且 combined_data_nd[i] 与 combined_labels[i] 严格对应。
    combined_labels : np.ndarray
        与 combined_data_nd 对应的标签数组，dtype=object。
    """
    # --- 输入检查与标准化为 list of 1D numpy arrays ---
    if isinstance(data, np.ndarray) and data.ndim == 2:
        # 等长信号矩阵 -> 转成 list（保留数值类型）
        data_list = [row.copy() for row in data]
    else:
        # 假设为序列，可含 numpy 数组或可转换为 numpy
        data_list = [np.asarray(x).copy() for x in data]

    n = len(data_list)
    if n == 0:
        raise ValueError("输入 data 为空。")
    if len(labels) != n:
        raise ValueError("data 和 labels 长度不一致。")

    labels_arr = np.array(labels, dtype=object)

    # 建立每类的样本索引池
    from collections import defaultdict, Counter
    pool_idx = defaultdict(list)
    for idx, lab in enumerate(labels_arr):
        pool_idx[lab].append(idx)

    # 计算每类数量
    unique, counts = np.unique(labels_arr, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    # 如果 target_per_class 未指定，则自动设为样本最多的类的数量
    if target_per_class is None:
        max_count = int(np.max(counts)) if len(counts) > 0 else 0
        target_per_class_effective = max_count
    else:
        # 若用户传入非正整数，仍回退到自动策略
        try:
            if int(target_per_class) <= 0:
                target_per_class_effective = int(np.max(counts))
            else:
                target_per_class_effective = int(target_per_class)
        except Exception:
            target_per_class_effective = int(np.max(counts))

    # 使用 numpy 的 new Generator
    rng = np.random.default_rng(random_state)

    # 计算需要增广数量（基于 effective target）
    need_augment = {cls: max(0, target_per_class_effective - counts_dict.get(cls, 0)) for cls in unique}

    aug_data = []
    aug_labels = []

    for cls, n_needed in need_augment.items():
        if n_needed <= 0:
            continue
        pool = pool_idx.get(cls, [])
        if len(pool) == 0:
            # 无原始样本，无法合成
            print(f"警告：类别 {cls} 在原始数据中不存在，无法增广。")
            continue
        if len(pool) == 1:
            # 提示单样本自混合
            print(f"提示：类别 {cls} 只有 1 个原始样本，增广将以同一条信号自混合方式进行。")

        for k in range(n_needed):
            # 从池中随机选择两条（可重复采样）
            ia = rng.choice(pool)
            ib = rng.choice(pool)

            xa = data_list[ia]
            xb = data_list[ib]

            # 对齐长度：截断到最短长度
            L = min(len(xa), len(xb))
            if L == 0:
                # 跳过空信号
                print(f"警告：在类别 {cls} 的候选样本中存在空信号，跳过一次合成。")
                continue
            xa_ = xa[:L]
            xb_ = xb[:L]

            # 随机0/1掩码（mix_ratio 的比例来自 xa，其余来自 xb）
            mask = rng.random(L) < mix_ratio
            x_new = np.where(mask, xa_, xb_)

            # 可选微小高斯噪声
            if add_small_noise:
                std_x = np.std(x_new)
                sigma = (std_x if std_x > 0 else 1.0) * noise_scale
                x_new = x_new + rng.normal(0.0, sigma, size=L)

            aug_data.append(np.asarray(x_new))
            aug_labels.append(cls)

    # 合并（原始保持在前，增广样本在后）
    combined_list = list(data_list) + aug_data
    combined_labels = np.concatenate([labels_arr, np.array(aug_labels, dtype=object)]) if len(aug_labels) > 0 else labels_arr

    # 尝试将 combined_list 转为 numpy ndarray 矩阵：
    # - 若所有信号等长 -> 2D 数组 (N_total, L)
    # - 否则 -> dtype=object 的 1D ndarray（每项为 1D ndarray），以确保顺序和对应关系不被破坏
    lengths = [len(x) for x in combined_list]
    all_equal_length = all(l == lengths[0] for l in lengths) if len(lengths) > 0 else True

    if all_equal_length:
        # 构造形状为 (N_total, L) 的数值型 ndarray
        try:
            combined_data_nd = np.stack(combined_list, axis=0)
        except Exception:
            # 若 stack 失败（例如数据类型不一致），退回为 object ndarray 以保证对应关系
            combined_data_nd = np.array(combined_list, dtype=object)
    else:
        # 长度不一，返回 object ndarray，每项为对应的 1D ndarray
        combined_data_nd = np.array(combined_list, dtype=object)

    # 输出增广后每类数量（便于检查）
    try:
        counter = Counter(combined_labels)
        print("均衡后各类数量：")
        for cls, cnt in sorted(counter.items(), key=lambda x: str(x[0])):
            print(f"  {cls}: {cnt}")
        print(f"(目标 target_per_class_effective = {target_per_class_effective})")
    except Exception:
        pass

    return combined_data_nd, combined_labels


@torch.no_grad()
def extract_global_features(dataset, device='cpu',latent_dim=1280):
    in_chans = dataset.data[0].shape[0]
    model = EEGDenoisingAutoencoder(in_channels=in_chans)
    model.load_state_dict(torch.load(cfgs.dae_last)['model_state'])
    model = model.to(device)
    model.eval()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    feat_row = []
    for i, batch in enumerate(train_loader, 1):
        x = batch.to(device)  # (B, C, N)
        feat_clean = model(x)
        feat_clean = feat_clean.numpy()
        feat_row.append(feat_clean)
    feat_data = np.concatenate(feat_row,axis=0)
    return feat_data

def start_gac(ds=None):
    from utils.loading_feature import load_2b,load_2a
    if not ds:
        ds = load_2a()
    feat_data = extract_global_features(ds,latent_dim=128)
    print(feat_data.shape)
    #主成分分析
    nc = 6
    pca = PCA(n_components=nc)
    feat_reduced = pca.fit_transform(feat_data)
    #加载特征和标签
    dataset = pd.read_csv(cfgs.feats).iloc[:, 1:]
    feats = dataset.drop(columns=['labels'])
    labels = dataset['labels']
    #增广数据
    syn_data = np.concatenate([feats.values,feat_reduced],axis=-1)
    syn_data,syn_labels = augment_time_series(syn_data,labels)
    #调整index
    cols = feats.columns
    new_cols = [f'D{i}' for i in range(1, nc+1)]
    cols = list(cols)+new_cols
    #保存为csv
    syn_data = pd.DataFrame(syn_data, columns=cols)
    syn_data['labels'] = labels
    syn_data.to_csv(cfgs.feats_syn)

if __name__ == "__main__":
    start_gac()