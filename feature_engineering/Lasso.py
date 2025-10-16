import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from config import cfg

# 稀疏回归（LassoCV 自动调节正则化参数）
sparse_selector = LassoCV(cv=5, random_state=42, max_iter=5000)

# 载入配置与数据
cfgs = cfg().get_args()

if __name__ == "__main__":
    # 读取数据
    dataset = pd.read_csv(cfgs.feats).iloc[:, 1:]
    feats = dataset.drop(columns=['labels'])
    labels = dataset['labels']
    print(feats.columns)

    # 拟合 Lasso（稀疏回归）
    sparse_selector.fit(feats, labels)

    # 获取被选中的特征（非零系数）
    chosen_feature_mask = sparse_selector.coef_ != 0
    chosen_feature_names = feats.columns[chosen_feature_mask]
    print("选择的特征：")
    print(chosen_feature_names)

    # 使用选择的特征构造最终特征矩阵
    feats_chosen = feats[chosen_feature_names]

    # 标准化
    standardizer = StandardScaler()
    standardizer.fit(feats_chosen)
    feats_scaled = standardizer.transform(feats_chosen)

    # 保存 csv
    feats_scaled = pd.DataFrame(feats_scaled, columns=chosen_feature_names)
    feats_scaled['labels'] = labels.to_numpy()
    feats_scaled.to_csv(cfgs.feats_scaled)

    # 输出维度信息
    print("原始特征维度：", feats.shape[1])
    print("筛选后特征维度：", feats_scaled.shape[1])
