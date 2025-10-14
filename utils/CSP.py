import numpy as np
from scipy.linalg import eigh
import torch
import warnings
from sklearn.preprocessing import LabelEncoder


class MultiClassCSPFeatureExtractor:
    def __init__(self, n_components=6, regularization=True, reg_param=1e-6,
                 max_iter=1000, tol=1e-6, diag_method='jade'):
        """
        多类CSP特征提取器（基于联合近似对角化）

        参数:
        n_components: int, CSP滤波器的数量
        regularization: bool, 是否启用正则化
        reg_param: float, 正则化参数
        max_iter: int, 联合对角化的最大迭代次数
        tol: float, 联合对角化的收敛容差
        diag_method: str, 对角化方法 ('jade', 'uwedge')
        """
        self.n_components = n_components
        self.regularization = regularization
        self.reg_param = reg_param
        self.max_iter = max_iter
        self.tol = tol
        self.diag_method = diag_method

        self.csp_filters = None
        self.label_encoder = None
        self.n_classes = None
        self.class_covariances = None
        self.whitening_matrix = None
        self.eigenvalues = None
    def filter(self):
        return self.csp_filters
    def fit(self, eeg_data, labels):
        """
        训练多类CSP滤波器

        参数:
        eeg_data: numpy数组，形状为 (n_trials, n_channels, n_samples)
        labels: numpy数组，形状为 (n_trials,)，包含类别标签
        """
        eeg_data = np.array(eeg_data)
        labels = np.array(labels)

        # 数据形状验证
        if len(eeg_data.shape) != 3:
            raise ValueError(f"EEG数据应为3维 (n_trials, n_channels, n_samples)，当前为{eeg_data.shape}")

        if len(labels.shape) != 1:
            raise ValueError("标签应为1维数组")

        if eeg_data.shape[0] != labels.shape[0]:
            raise ValueError("EEG数据和标签的试验数量不匹配")

        # 标签编码
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.n_classes = len(self.label_encoder.classes_)

        # 检查每个类别的样本数
        unique_labels, counts = np.unique(encoded_labels, return_counts=True)
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            if count < 2:
                warnings.warn(f"类别 {self.label_encoder.classes_[i]} 的样本数少于2个，可能影响CSP效果")

        n_trials, n_channels, n_samples = eeg_data.shape

        # 计算每个类别的协方差矩阵
        self.class_covariances = []
        for class_idx in range(self.n_classes):
            class_data = eeg_data[encoded_labels == class_idx]
            if len(class_data) > 0:
                class_cov = self._compute_covariance_matrix(class_data)
                self.class_covariances.append(class_cov)
            else:
                raise ValueError(f"类别 {class_idx} 没有数据")

        # 如果只有两类，使用传统的二分类CSP
        if self.n_classes == 2:
            self.csp_filters = self._fit_binary_csp()
        else:
            # 使用联合近似对角化的多类CSP
            self.csp_filters = self._fit_multiclass_csp()

        return self.csp_filters

    def _fit_binary_csp(self):
        """传统的二分类CSP"""
        cov_0, cov_1 = self.class_covariances[0], self.class_covariances[1]

        # 计算复合协方差矩阵
        cov_composite = cov_0 + cov_1

        # 应用正则化
        if self.regularization:
            cov_composite = self._apply_regularization(cov_composite, "composite")
            cov_0 = self._apply_regularization(cov_0, "class_0")
            cov_1 = self._apply_regularization(cov_1, "class_1")
        else:
            cov_composite = self._ensure_basic_stability(cov_composite)
            cov_0 = self._ensure_basic_stability(cov_0)
            cov_1 = self._ensure_basic_stability(cov_1)

        # 特征值分解和白化
        try:
            eigenvalues, eigenvectors = eigh(cov_composite)
        except Exception as e:
            warnings.warn(f"协方差矩阵特征值分解失败: {e}")
            cov_composite = self._apply_regularization(cov_composite, "emergency", self.reg_param * 10)
            eigenvalues, eigenvectors = eigh(cov_composite)

        # 处理小的特征值
        min_eigenvalue = 1e-12 if self.regularization else 1e-8
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 计算白化矩阵
        self.whitening_matrix = np.dot(eigenvectors, np.diag(eigenvalues ** -0.5))
        self.eigenvalues = eigenvalues

        # 白化协方差矩阵
        S0_white = np.dot(np.dot(self.whitening_matrix.T, cov_0), self.whitening_matrix)
        S1_white = np.dot(np.dot(self.whitening_matrix.T, cov_1), self.whitening_matrix)

        # 应用正则化到白化后的矩阵
        if self.regularization:
            S0_white = self._apply_regularization(S0_white, "S0_white")
            S1_white = self._apply_regularization(S1_white, "S1_white")

        # 广义特征值分解
        try:
            eigenvalues_csp, eigenvectors_csp = eigh(S0_white, S0_white + S1_white)
        except Exception as e:
            warnings.warn(f"广义特征值分解失败: {e}")
            S0_white = self._apply_regularization(S0_white, "emergency", self.reg_param * 10)
            S1_white = self._apply_regularization(S1_white, "emergency", self.reg_param * 10)
            eigenvalues_csp, eigenvectors_csp = eigh(S0_white, S0_white + S1_white)

        idx = np.argsort(eigenvalues_csp)[::-1]
        eigenvectors_csp = eigenvectors_csp[:, idx]

        # 计算CSP滤波器
        csp_filters = np.dot(self.whitening_matrix, eigenvectors_csp)

        # 选择滤波器
        n_select = min(self.n_components // 2, csp_filters.shape[1] // 2)
        selected_filters = np.column_stack([
            csp_filters[:, :n_select],
            csp_filters[:, -n_select:]
        ])

        return selected_filters

    def _fit_multiclass_csp(self):
        """基于联合近似对角化的多类CSP"""
        # 计算平均协方差矩阵
        avg_cov = np.mean(self.class_covariances, axis=0)

        # 应用正则化
        if self.regularization:
            avg_cov = self._apply_regularization(avg_cov, "average_cov")
            regularized_class_covs = [self._apply_regularization(cov, f"class_{i}")
                                      for i, cov in enumerate(self.class_covariances)]
        else:
            avg_cov = self._ensure_basic_stability(avg_cov)
            regularized_class_covs = [self._ensure_basic_stability(cov)
                                      for cov in self.class_covariances]

        # 特征值分解和白化
        try:
            eigenvalues, eigenvectors = eigh(avg_cov)
        except Exception as e:
            warnings.warn(f"平均协方差矩阵特征值分解失败: {e}")
            avg_cov = self._apply_regularization(avg_cov, "emergency", self.reg_param * 10)
            eigenvalues, eigenvectors = eigh(avg_cov)

        # 处理小的特征值
        min_eigenvalue = 1e-12 if self.regularization else 1e-8
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 计算白化矩阵
        self.whitening_matrix = np.dot(eigenvectors, np.diag(eigenvalues ** -0.5))
        self.eigenvalues = eigenvalues

        # 白化所有类别的协方差矩阵
        whitened_covs = []
        for cov in regularized_class_covs:
            whitened_cov = np.dot(np.dot(self.whitening_matrix.T, cov), self.whitening_matrix)
            whitened_covs.append(whitened_cov)

        # 联合近似对角化
        if self.diag_method == 'jade':
            diag_matrix = self._jade_diagonalization(whitened_covs)
        else:
            diag_matrix = self._uwedge_diagonalization(whitened_covs)

        # 计算最终的CSP滤波器
        csp_filters = np.dot(self.whitening_matrix, diag_matrix)

        # 选择最优滤波器
        selected_filters = self._select_best_filters(csp_filters, regularized_class_covs)

        return selected_filters

    def _jade_diagonalization(self, matrices):
        """JADE算法进行联合对角化"""
        n_matrices = len(matrices)
        n_channels = matrices[0].shape[0]

        # 初始化对角化矩阵
        V = np.eye(n_channels)

        for iteration in range(self.max_iter):
            old_V = V.copy()

            # 对每一对通道进行Jacobi旋转
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    # 计算旋转角度
                    angle = self._compute_jacobi_angle(matrices, V, i, j)

                    # 构造旋转矩阵
                    G = np.eye(n_channels)
                    c, s = np.cos(angle), np.sin(angle)
                    G[i, i] = c
                    G[j, j] = c
                    G[i, j] = -s
                    G[j, i] = s

                    # 更新V
                    V = np.dot(V, G)

            # 检查收敛
            if np.linalg.norm(V - old_V) < self.tol:
                break

        return V

    def _compute_jacobi_angle(self, matrices, V, i, j):
        """计算Jacobi旋转角度"""
        h = 0
        g = 0

        for matrix in matrices:
            # 当前对角化状态
            diag_matrix = np.dot(np.dot(V.T, matrix), V)

            # 计算off-diagonal元素
            off_diag_ij = diag_matrix[i, j]
            off_diag_ji = diag_matrix[j, i]
            diag_ii = diag_matrix[i, i]
            diag_jj = diag_matrix[j, j]

            h += off_diag_ij + off_diag_ji
            g += diag_ii - diag_jj

        # 计算旋转角度
        if abs(h) < 1e-12:
            return 0
        else:
            return 0.5 * np.arctan2(2 * h, g)

    def _uwedge_diagonalization(self, matrices):
        """UWEDGE算法进行联合对角化"""
        n_matrices = len(matrices)
        n_channels = matrices[0].shape[0]

        # 初始化
        V = np.eye(n_channels)

        for iteration in range(self.max_iter):
            old_V = V.copy()

            # 计算当前的对角化程度
            current_matrices = [np.dot(np.dot(V.T, matrix), V) for matrix in matrices]

            # 计算梯度
            gradient = np.zeros((n_channels, n_channels))
            for matrix in current_matrices:
                off_diag = matrix - np.diag(np.diag(matrix))
                gradient += np.dot(off_diag, matrix.T) - np.dot(matrix, off_diag.T)

            # 更新V
            step_size = 0.1 / (iteration + 1)
            V = V - step_size * np.dot(V, gradient)

            # 正交化
            U, _, Vt = np.linalg.svd(V)
            V = np.dot(U, Vt)

            # 检查收敛
            if np.linalg.norm(V - old_V) < self.tol:
                break

        return V

    def _select_best_filters(self, csp_filters, class_covs):
        """选择最优的CSP滤波器"""
        n_channels = csp_filters.shape[0]
        n_filters = csp_filters.shape[1]

        # 计算每个滤波器的判别能力
        discriminative_power = np.zeros(n_filters)

        for f in range(n_filters):
            filter_vector = csp_filters[:, f]

            # 计算该滤波器下各类别的方差
            class_variances = []
            for cov in class_covs:
                variance = np.dot(np.dot(filter_vector.T, cov), filter_vector)
                class_variances.append(variance)

            # 计算判别能力（类间方差与类内方差的比值）
            class_variances = np.array(class_variances)
            between_class_var = np.var(class_variances)
            within_class_var = np.mean(class_variances)

            if within_class_var > 1e-12:
                discriminative_power[f] = between_class_var / within_class_var
            else:
                discriminative_power[f] = 0

        # 选择判别能力最强的滤波器
        selected_indices = np.argsort(discriminative_power)[::-1][:self.n_components]
        selected_filters = csp_filters[:, selected_indices]

        return selected_filters

    def transform(self, eeg_data):
        """应用CSP滤波器进行特征提取"""
        if self.csp_filters is None:
            raise ValueError("CSP滤波器未训练，请先调用fit方法")

        eeg_data = np.array(eeg_data)

        # 检查输入数据形状
        if len(eeg_data.shape) not in [2, 3]:
            raise ValueError("输入数据应为2维或3维")

        # 检查输入数据
        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
            eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
            warnings.warn("输入数据包含无效值，已修复")

        # 应用CSP滤波器
        if len(eeg_data.shape) == 3:  # 多个trial
            n_trials, n_channels, n_samples = eeg_data.shape

            # 检查通道数是否匹配
            if n_channels != self.csp_filters.shape[0]:
                raise ValueError(f"通道数不匹配：训练时{self.csp_filters.shape[0]}，当前{n_channels}")

            csp_features_list = []

            for trial in range(n_trials):
                trial_data = eeg_data[trial]
                filtered_data = np.dot(self.csp_filters.T, trial_data)

                # 计算方差特征
                variances = np.var(filtered_data, axis=1)

                # 确保方差为正数
                min_variance = 1e-12 if self.regularization else 1e-8
                variances = np.maximum(variances, min_variance)

                # 对数变换
                log_variances = np.log(variances)

                # 检查结果
                if np.any(np.isnan(log_variances)) or np.any(np.isinf(log_variances)):
                    log_variances = np.nan_to_num(log_variances, nan=0.0, posinf=0.0, neginf=0.0)
                    warnings.warn(f"Trial {trial} 的CSP特征包含无效值，已修复")

                csp_features_list.append(log_variances)

            csp_features = np.array(csp_features_list)

        else:  # 单个trial
            if eeg_data.shape[0] != self.csp_filters.shape[0]:
                raise ValueError(f"通道数不匹配：训练时{self.csp_filters.shape[0]}，当前{eeg_data.shape[0]}")

            filtered_data = np.dot(self.csp_filters.T, eeg_data)

            # 计算方差特征
            variances = np.var(filtered_data, axis=1)

            # 确保方差为正数
            min_variance = 1e-12 if self.regularization else 1e-8
            variances = np.maximum(variances, min_variance)

            # 对数变换
            log_variances = np.log(variances)

            # 检查结果
            if np.any(np.isnan(log_variances)) or np.any(np.isinf(log_variances)):
                log_variances = np.nan_to_num(log_variances, nan=0.0, posinf=0.0, neginf=0.0)
                warnings.warn("CSP特征包含无效值，已修复")

            csp_features = log_variances

        csp_features = torch.from_numpy(csp_features).clone().float()
        return csp_features

    def _compute_covariance_matrix(self, data):
        """计算归一化的协方差矩阵"""
        n_trials, n_channels, n_samples = data.shape
        cov_matrix = np.zeros((n_channels, n_channels))
        valid_trials = 0

        for trial in range(n_trials):
            trial_data = data[trial]

            # 检查trial数据
            if np.any(np.isnan(trial_data)) or np.any(np.isinf(trial_data)):
                trial_data = np.nan_to_num(trial_data, nan=0.0, posinf=0.0, neginf=0.0)
                warnings.warn(f"Trial {trial} 包含无效值，已修复")

            # 计算协方差矩阵
            try:
                trial_cov = np.cov(trial_data)
            except Exception as e:
                warnings.warn(f"Trial {trial} 协方差计算失败: {e}")
                continue

            # 检查协方差矩阵
            if np.any(np.isnan(trial_cov)) or np.any(np.isinf(trial_cov)):
                trial_cov = np.nan_to_num(trial_cov, nan=0.0, posinf=0.0, neginf=0.0)
                warnings.warn(f"Trial {trial} 协方差矩阵包含无效值，已修复")

            # 归一化
            trace = np.trace(trial_cov)
            if trace > 1e-12:
                trial_cov = trial_cov / trace
                cov_matrix += trial_cov
                valid_trials += 1
            else:
                warnings.warn(f"Trial {trial} 协方差矩阵trace过小，跳过")

        if valid_trials == 0:
            raise ValueError("没有有效的trial用于计算协方差矩阵")

        return cov_matrix / valid_trials

    def _apply_regularization(self, matrix, matrix_name="matrix", reg_param=None):
        """应用正则化到矩阵"""
        if reg_param is None:
            reg_param = self.reg_param

        # 清理无效值
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # 确保矩阵是对称的
        matrix = (matrix + matrix.T) / 2

        # 计算条件数以确定是否需要更强的正则化
        try:
            cond_num = np.linalg.cond(matrix)
            if cond_num > 1e12:
                adapted_reg_param = reg_param * np.sqrt(cond_num / 1e12)
                warnings.warn(f"{matrix_name} 条件数过大 ({cond_num:.2e})，使用自适应正则化参数 {adapted_reg_param:.2e}")
                reg_param = adapted_reg_param
        except:
            pass

        # 添加正则化项
        matrix += reg_param * np.eye(matrix.shape[0])

        # 收缩估计
        if self.reg_param > 1e-4:
            shrinkage_factor = min(0.1, self.reg_param * 10)
            identity_matrix = np.eye(matrix.shape[0])
            matrix = (1 - shrinkage_factor) * matrix + shrinkage_factor * identity_matrix

        return matrix

    def _ensure_basic_stability(self, matrix):
        """确保矩阵的基本数值稳定性"""
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = (matrix + matrix.T) / 2
        return matrix

    def fit_transform(self, eeg_data, labels):
        """训练并变换数据"""
        self.fit(eeg_data, labels)
        return self.transform(eeg_data)

    def get_filter_info(self):
        """获取滤波器信息"""
        if self.csp_filters is None:
            return "CSP滤波器未训练"

        info = {
            "n_channels": self.csp_filters.shape[0],
            "n_components": self.csp_filters.shape[1],
            "n_classes": self.n_classes,
            "class_labels": self.label_encoder.classes_ if self.label_encoder else None,
            "filter_shape": self.csp_filters.shape,
            "regularization": self.regularization,
            "reg_param": self.reg_param,
            "diag_method": self.diag_method
        }
        return info

    def set_regularization(self, regularization=True, reg_param=1e-6):
        """设置正则化参数"""
        self.regularization = regularization
        self.reg_param = reg_param

        if self.csp_filters is not None:
            warnings.warn("正则化参数已更改，建议重新训练CSP滤波器")


# 保持向后兼容的二分类CSP类
class CSPFeatureExtractor(MultiClassCSPFeatureExtractor):
    """向后兼容的二分类CSP特征提取器"""

    def __init__(self, n_components=6, regularization=True, reg_param=1e-6):
        super().__init__(n_components=n_components, regularization=regularization, reg_param=reg_param)
        self.label_map = None

    def fit(self, eeg_data, labels):
        """训练CSP滤波器（保持向后兼容）"""
        # 检查是否为二分类问题
        unique_labels = np.unique(labels)
        if len(unique_labels) > 2:
            warnings.warn("检测到多类数据，建议使用MultiClassCSPFeatureExtractor")

        # 调用父类的fit方法
        result = super().fit(eeg_data, labels)

        # 为向后兼容设置label_map
        if self.label_encoder is not None and len(self.label_encoder.classes_) == 2:
            self.label_map = {
                self.label_encoder.classes_[0]: 0,
                self.label_encoder.classes_[1]: 1
            }

        return result


# 便捷函数
def extract_csp_features(eeg_data, labels, n_components=6, regularization=True, reg_param=1e-6):
    """提取CSP特征的便捷函数（自动检测是否为多类问题）"""
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 2:
        # 二分类问题
        csp = CSPFeatureExtractor(n_components=n_components,
                                  regularization=regularization,
                                  reg_param=reg_param)
    else:
        # 多分类问题
        csp = MultiClassCSPFeatureExtractor(n_components=n_components,
                                            regularization=regularization,
                                            reg_param=reg_param)

    return csp.fit_transform(eeg_data, labels)


def apply_csp_to_new_data(csp_extractor, new_eeg_data):
    """应用已训练的CSP到新数据"""
    return csp_extractor.transform(new_eeg_data)


def create_multiclass_csp(n_components=6, reg_strength="medium", diag_method="jade"):
    """创建多类CSP实例"""
    reg_params = {
        "none": (False, 0),
        "light": (True, 1e-8),
        "medium": (True, 1e-6),
        "strong": (True, 1e-4)
    }

    if reg_strength not in reg_params:
        raise ValueError(f"reg_strength必须是以下之一: {list(reg_params.keys())}")

    regularization, reg_param = reg_params[reg_strength]

    return MultiClassCSPFeatureExtractor(
        n_components=n_components,
        regularization=regularization,
        reg_param=reg_param,
        diag_method=diag_method
    )


def create_csp_with_regularization(reg_strength="medium", n_components=6):
    """创建具有预设正则化强度的CSP实例（保持向后兼容）"""
    reg_params = {
        "none": (False, 0),
        "light": (True, 1e-8),
        "medium": (True, 1e-6),
        "strong": (True, 1e-4)
    }

    if reg_strength not in reg_params:
        raise ValueError(f"reg_strength必须是以下之一: {list(reg_params.keys())}")

    regularization, reg_param = reg_params[reg_strength]
    return CSPFeatureExtractor(n_components=n_components,
                               regularization=regularization,
                               reg_param=reg_param)