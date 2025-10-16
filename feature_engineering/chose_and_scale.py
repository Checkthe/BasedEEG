import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from config import cfg

def start_cas():
    # 基学习器（随机森林）
    base_learner = RandomForestClassifier(n_estimators=200, random_state=42)

    # 构造递归特征消除器（RFE），保留特征数可调整
    rfe_engine = RFE(base_learner, n_features_to_select=60, step=60)

    # 载入配置与数据
    cfgs = cfg().get_args()
    dataset = pd.read_csv(cfgs.feats_syn).iloc[:, 1:]
    feats = dataset.drop(columns=['labels'])
    labels = dataset['labels']
    print(feats.columns)
    # 拟合 RFE（使用筛选后的源域特征与标签）
    rfe_engine = rfe_engine.fit(feats, labels)

    # 获取被 RFE 选中的特征名
    chosen_feature_names = feats.columns[rfe_engine.support_]
    chosen_feature_names = [f'D{i}' for i in range(1, 7)]
    print("选择的特征：")
    print(chosen_feature_names)

    # 使用选择的特征构造最终特征矩阵
    feats_chosen = feats[chosen_feature_names]

    # 标准化
    standardizer = StandardScaler()
    standardizer.fit(feats_chosen)
    feats_scaled = standardizer.transform(feats_chosen)

    #保存csv
    feats_scaled = pd.DataFrame(feats_scaled,columns=chosen_feature_names)
    feats_scaled['labels'] = labels.to_numpy()
    feats_scaled.to_csv(cfgs.feats_scaled)

    # 输出维度信息
    print("原始特征维度：", feats.shape[1])
    print("筛选后特征维度：", feats_scaled.shape[1])

if __name__ == "__main__":
    start_cas()