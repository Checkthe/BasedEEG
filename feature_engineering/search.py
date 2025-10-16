import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from config import cfg

# 读取配置与数据
cfgs = cfg().get_args()

# 读数据
dataset = pd.read_csv(cfgs.feats_scaled)
source_features = dataset.iloc[:,:-1]
labels = dataset.iloc[:,-1]

# ========== 数据划分 ==========
X_train, X_test, y_train, y_test = train_test_split(
    source_features, labels, test_size=0.3, random_state=42, stratify=labels
)

# 训练/验证（此处直接使用上面划分）
X_train_inner, X_val, y_train_inner, y_val = X_train, X_test, y_train, y_test

# ========== 超参数空间定义 ==========
HYPER_SPACE = {
    "n_estimators": (50, 1000),
    "max_depth": (3, 50),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 20)
}

def vector_to_rf_params(position_vector):
    """将连续向量解码为随机森林的超参数字典"""
    ne_lo, ne_hi = HYPER_SPACE["n_estimators"]
    md_lo, md_hi = HYPER_SPACE["max_depth"]
    mss_lo, mss_hi = HYPER_SPACE["min_samples_split"]
    msl_lo, msl_hi = HYPER_SPACE["min_samples_leaf"]

    params = {
        "n_estimators": int(round(np.clip(position_vector[0], ne_lo, ne_hi))),
        "max_depth": int(round(np.clip(position_vector[1], md_lo, md_hi))),
        "min_samples_split": int(round(np.clip(position_vector[2], mss_lo, mss_hi))),
        "min_samples_leaf": int(round(np.clip(position_vector[3], msl_lo, msl_hi))),
        "random_state": 42,
        "n_jobs": -1,
    }
    # 修正边界关系（min_samples_split 不应小于 min_samples_leaf）
    params["min_samples_split"] = max(params["min_samples_split"], params["min_samples_leaf"])
    return params

def evaluate_candidate(position_vector):
    """在验证集上评估一个候选超参数（返回 F1_macro，越大越好）"""
    params = vector_to_rf_params(position_vector)
    clf = RandomForestClassifier(**params)
    clf.fit(X_train_inner, y_train_inner)
    preds = clf.predict(X_val)
    return f1_score(y_val, preds, average="macro")

# ========== PSO 主流程参数 ==========
DIM = 4
NUM_PARTICLES = 20
NUM_ITER = 25
W_INERTIA = 0.72
C1 = 1.49
C2 = 1.49
rng = np.random.default_rng(2025)

LOWS = np.array([
    HYPER_SPACE["n_estimators"][0],
    HYPER_SPACE["max_depth"][0],
    HYPER_SPACE["min_samples_split"][0],
    HYPER_SPACE["min_samples_leaf"][0]
])
HIGHS = np.array([
    HYPER_SPACE["n_estimators"][1],
    HYPER_SPACE["max_depth"][1],
    HYPER_SPACE["min_samples_split"][1],
    HYPER_SPACE["min_samples_leaf"][1]
])

positions = rng.uniform(LOWS, HIGHS, size=(NUM_PARTICLES, DIM))
velocities = rng.uniform(-np.abs(HIGHS - LOWS), np.abs(HIGHS - LOWS), size=(NUM_PARTICLES, DIM)) * 0.1

# 初始评估
personal_best_pos = positions.copy()
personal_best_score = np.array([evaluate_candidate(p) for p in positions])

global_best_idx = int(np.argmax(personal_best_score))
global_best_pos = personal_best_pos[global_best_idx].copy()
global_best_score = float(personal_best_score[global_best_idx])

print(f"[PSO] 初始最优 F1_macro = {global_best_score:.4f}, 超参数 = {vector_to_rf_params(global_best_pos)}")

start_time = time.time()
for iter_idx in range(1, NUM_ITER + 1):
    r1 = rng.random((NUM_PARTICLES, DIM))
    r2 = rng.random((NUM_PARTICLES, DIM))

    velocities = (
        W_INERTIA * velocities
        + C1 * r1 * (personal_best_pos - positions)
        + C2 * r2 * (global_best_pos - positions)
    )
    positions = positions + velocities
    # 边界裁剪
    positions = np.minimum(np.maximum(positions, LOWS), HIGHS)

    scores = np.array([evaluate_candidate(p) for p in positions])

    improved_mask = scores > personal_best_score
    if np.any(improved_mask):
        personal_best_pos[improved_mask] = positions[improved_mask]
        personal_best_score[improved_mask] = scores[improved_mask]

    if personal_best_score.max() > global_best_score:
        global_best_idx = int(np.argmax(personal_best_score))
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_score = float(personal_best_score[global_best_idx])

    if iter_idx % 5 == 0 or iter_idx == NUM_ITER:
        print(f"[PSO] 迭代 {iter_idx:02d}/{NUM_ITER}，当前最优 F1_macro = {global_best_score:.4f}")

end_time = time.time()
print(f"[PSO] 完成。耗时 {end_time - start_time:.1f}s")
best_hyperparams = vector_to_rf_params(global_best_pos)
print("[PSO] 最优超参数：", best_hyperparams)

# ========== 用最优超参数在训练集上重训并在测试集评估 ==========
best_clf = RandomForestClassifier(**best_hyperparams)
best_clf.fit(X_train, y_train)

y_pred_test = best_clf.predict(X_test)
# 若需要概率输出
try:
    y_proba_test = best_clf.predict_proba(X_test)
except Exception:
    y_proba_test = None

f1_macro_on_test = f1_score(y_test, y_pred_test, average="macro")
print(f"[PSO] 最优模型在测试集上的 F1_macro：{f1_macro_on_test:.4f}")
