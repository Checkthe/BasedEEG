import numpy as np
from scipy.linalg import null_space
import torch

#求解线性方程组
def solve_homogeneous(A: np.ndarray, n: int) -> np.ndarray:
    """
    求解 AX = 0 的线性无关 n 个解。

    参数:
    - A: numpy.ndarray，形状 (N, C)
    - n: int，所需线性无关解的数量

    返回:
    - X: numpy.ndarray，形状 (C, n)，AX=0，X 的列线性无关
    """
    N, C = A.shape
    null_basis = null_space(A)  # 形状为 (C, k), k = C - rank(A)
    null_dim = null_basis.shape[1]
    if n > null_dim:
        raise ValueError(f"要求 {n} 个线性无关解，但零空间维度仅为 {null_dim}，无法满足。")

    # 随机正交变换后取前 n 个线性无关列
    Q, _ = np.linalg.qr(np.random.randn(null_dim, null_dim))  # 正交矩阵 (null_dim, null_dim)
    X = null_basis @ Q[:, :n]  # (C, null_dim) × (null_dim, n) = (C, n)
    return X

def approx_nullspace(A: np.ndarray, n: int) -> np.ndarray:
    """
    返回使 AX ≈ 0 的近似解 X ∈ ℝ^{C×n}，X 的列线性无关。

    参数：
    - A: (N, C) numpy array
    - n: 所需近似 null space 解的个数

    返回：
    - X: (C, n) numpy array，满足 AX ≈ 0
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # 最小的 n 个奇异值对应的右奇异向量（V 的最后 n 列）
    X = Vt[-n:].T  # (C, n)
    return X

from scipy.optimize import minimize

def approx_nullspace_optimum(A,seed=42):
    C = A.shape[1]
    np.random.seed(seed)
    def objective(x):
        X = x.reshape(C, C)
        return np.linalg.norm(A @ X, ord='fro')**2

    x0 = np.random.randn(C, C).flatten()
    result = minimize(objective, x0, method='L-BFGS-B')
    X = result.x.reshape(C, C)
    return X


#计算协方差矩阵
def compute_csp_covariance(eeg_data: torch.Tensor) -> torch.Tensor:
    """
    计算 EEG 信号的协方差矩阵（按 CSP 方法）。

    参数:
        eeg_data: torch.Tensor, 形状为 (batch_size, channels, n_points)

    返回:
        cov_matrices: torch.Tensor, 形状为 (batch_size, channels, channels)
    """
    batch_size, channels, n_points = eeg_data.shape

    # 矩阵乘法计算协方差并进行trace归一化
    cov_matrices = torch.zeros((batch_size, channels, channels), dtype=eeg_data.dtype, device=eeg_data.device)

    for i in range(batch_size):
        X = eeg_data[i]  # 形状 (channels, n_points)
        S = X @ X.T  # (channels, channels)
        S = S / torch.trace(S)
        cov_matrices[i] = S

    return cov_matrices
