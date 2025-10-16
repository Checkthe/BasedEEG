import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from config import cfg

def start_RFC():
    # ---------- 配置 ----------
    cfgs = cfg().get_args()
    DATA_PATH = cfgs.feats_scaled
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    RF_PARAMS = {
        'n_estimators': 677,
        'max_depth': 38,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    SAVE_FIGS = False
    OUT_DIR = getattr(cfgs, 'out_dir', '.')  # 如果配置中没有 out_dir，则默认当前目录
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 读取数据 ----------
    df = pd.read_csv(DATA_PATH).iloc[:, 1:]
    feats = df.drop(columns=['labels'])
    labels = df['labels'].values
    # ---------- 划分训练/测试集 ----------
    X_train, X_val, y_train, y_val = train_test_split(
        feats, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    # ---------- 构建并训练分类器 ----------
    model_rf = RandomForestClassifier(**RF_PARAMS)
    model_rf.fit(X_train, y_train)

    # ---------- 预测与概率 ----------
    y_pred = model_rf.predict(X_val)

    # 有些分类器没有 predict_proba（但 RandomForest 有），这里做稳健处理
    if hasattr(model_rf, "predict_proba"):
        y_score = model_rf.predict_proba(X_val)
    elif hasattr(model_rf, "decision_function"):
        # decision_function 返回 shape (n_samples, n_classes) 或 (n_samples,)
        dec = model_rf.decision_function(X_val)
        # 若为一维（binary），将其转为两列概率样式（sigmoid）：
        if dec.ndim == 1:
            from scipy.special import expit
            probs_pos = expit(dec)
            y_score = np.vstack([1 - probs_pos, probs_pos]).T
        else:
            # 对多类做 softmax 近似
            exp_dec = np.exp(dec - np.max(dec, axis=1, keepdims=True))
            y_score = exp_dec / exp_dec.sum(axis=1, keepdims=True)
    else:
        raise RuntimeError("模型既不支持 predict_proba 也不支持 decision_function，无法计算概率用于 ROC。")

    # ---------- 准备类信息 ----------
    unique_classes = np.unique(labels)
    n_classes = unique_classes.shape[0]

    # 标签二值化（用于多类 ROC）
    y_val_binarized = label_binarize(y_val, classes=unique_classes)
    if n_classes == 2 and y_val_binarized.shape[1] == 1:
        # 二分类时 label_binarize 可能返回一列，统一为两列（neg,pos）
        y_val_binarized = np.hstack([1 - y_val_binarized, y_val_binarized])

    # ---------- 评估报告 ----------
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=[str(c) for c in unique_classes]))

    # ---------- 多类 ROC 曲线（每类 + micro/macro） ----------
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 6))

    # 存放每类 fpr/tpr/auc 用于后续输出
    per_class_results = {}

    # 计算每类 ROC
    for i, cls in enumerate(unique_classes):
        # y_score 列索引对应 class 顺序：sklearn 的 predict_proba 返回列顺序与 model.classes_ 一致
        # 为稳健起见，根据 model.classes_ 寻找列索引
        try:
            class_index = list(model_rf.classes_).index(cls)
        except ValueError:
            class_index = i  # 兜底：按 i 使用
        fpr, tpr, _ = roc_curve(y_val_binarized[:, i], y_score[:, class_index])
        roc_auc = auc(fpr, tpr)
        per_class_results[cls] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    # Micro-average ROC
    # 将所有类的二值化标签和分数展平后计算
    y_val_flat = y_val_binarized.ravel()
    # 对 y_score 也需要以相同顺序展平（按 model_rf.classes_ 列序）
    # 先构建按 unique_classes 顺序的列索引映射到 model_rf.classes_ 的列索引
    col_indices = [list(model_rf.classes_).index(c) for c in unique_classes]
    y_score_reordered = y_score[:, col_indices]
    y_score_flat = y_score_reordered.ravel()
    fpr_micro, tpr_micro, _ = roc_curve(y_val_flat, y_score_flat)
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle=':', linewidth=2.5, label=f"micro-average (AUC = {auc_micro:.3f})")

    # Macro-average ROC:
    # 先插值并计算平均 TPR 在统一的 FPR 网格上
    all_fpr = np.unique(np.concatenate([per_class_results[c]['fpr'] for c in unique_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in unique_classes:
        mean_tpr += np.interp(all_fpr, per_class_results[c]['fpr'], per_class_results[c]['tpr'])
    mean_tpr /= n_classes
    auc_macro = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, linestyle='--', linewidth=2.5, label=f"macro-average (AUC = {auc_macro:.3f})")

    # 对角基线
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if SAVE_FIGS:
        roc_path = os.path.join(OUT_DIR, "multiclass_roc.png")
        plt.savefig(roc_path, dpi=300)
        print(f"Saved ROC figure to: {roc_path}")
    plt.show()

    # ---------- 混淆矩阵（原始与归一化） ----------
    cm = confusion_matrix(y_val, y_pred, labels=unique_classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (Counts)")
    plt.tight_layout()
    if SAVE_FIGS:
        cm_path = os.path.join(OUT_DIR, "confusion_matrix_counts.png")
        plt.savefig(cm_path, dpi=300)
        print(f"Saved confusion matrix (counts) to: {cm_path}")
    plt.show()

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (Normalized by True Class)")
    plt.tight_layout()
    if SAVE_FIGS:
        cmn_path = os.path.join(OUT_DIR, "confusion_matrix_normalized.png")
        plt.savefig(cmn_path, dpi=300)
        print(f"Saved confusion matrix (normalized) to: {cmn_path}")
    plt.show()

    # ---------- 每类 AUC 输出 ----------
    print("Per-class AUCs:")
    for cls in unique_classes:
        print(f"  Class {cls}: AUC = {per_class_results[cls]['auc']:.4f}")
    print(f"Macro AUC: {auc_macro:.4f}, Micro AUC: {auc_micro:.4f}")

if __name__ == "__main__":
    start_RFC()