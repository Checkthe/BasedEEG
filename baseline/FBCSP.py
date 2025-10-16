import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    data: array (..., n_samples)
    returns filtered data with same shape
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # filtfilt along last axis
    return filtfilt(b, a, data, axis=-1)


def cov_normalized(trial):
    """trial: (n_channels, n_samples) -> normalized covariance matrix"""
    X = trial
    C = X @ X.T / X.shape[1]
    return C / np.trace(C)


def compute_average_covariances(X, y, classes):
    """
    X: list/array of trials shape (n_trials, n_channels, n_samples)
    y: (n_trials,)
    returns dict[class -> avg_cov matrix]
    """
    avg_cov = {}
    for c in classes:
        inds = np.where(y == c)[0]
        covs = np.stack([cov_normalized(X[i]) for i in inds], axis=0)
        avg_cov[c] = covs.mean(axis=0)
    return avg_cov


def csp_fit_binary(X, y, n_components=4):
    """
    Fit CSP for binary classes.
    Returns projection matrix W of shape (n_channels, n_components)
    The returned components are ordered: from most to least discriminative (so you can take first k).
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("csp_fit_binary only supports binary classification.")
    covs = compute_average_covariances(X, y, classes)
    S0 = covs[classes[0]]
    S1 = covs[classes[1]]
    # Solve generalized eigenvalue problem: S0 w = lambda (S0+S1) w
    R = S0 + S1
    # eigh returns eigenvals ascending
    eigvals, eigvecs = eigh(S0, R)
    # sort by |eig-0.5| descending or by eigenvalue distance from 0/1
    # classical approach: take eigenvectors associated with largest and smallest eigenvalues
    # eigvecs columns correspond to eigenvalues
    # sort indices by eigenvalue
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    # pick from both ends to form pairs
    n_ch = X.shape[1]
    if n_components % 2 != 0:
        raise ValueError("n_components should be even (pairs from both ends).")
    picks = []
    half = n_components // 2
    for i in range(half):
        picks.append(eigvecs[:, i])               # smallest
        picks.append(eigvecs[:, -1 - i])          # largest
    W = np.column_stack(picks)  # (n_channels, n_components)
    return W


def extract_logvar_features(X, W):
    """
    X: (n_trials, n_channels, n_samples)
    W: (n_channels, n_components)
    returns features: (n_trials, n_components) -> log var normalized
    """
    n_trials = X.shape[0]
    n_comp = W.shape[1]
    feats = np.zeros((n_trials, n_comp))
    for i in range(n_trials):
        projected = W.T @ X[i]  # (n_components, n_samples)
        var = np.var(projected, axis=1)
        # normalization often used: divide by sum of variances across components
        var = var / np.sum(var)
        feats[i] = np.log(var + 1e-10)
    return feats


class FBCSP_LDA(BaseEstimator, ClassifierMixin):
    """
    FBCSP + LDA pipeline for binary classification.
    Usage:
        model = FBCSP_LDA(fs=250, bands=[(8,12),(12,16),...], n_components=4)
        model.fit(X_train, y_train)  # X_train shape: (n_trials, n_channels, n_samples)
        preds = model.predict(X_test)
    Attributes after fit:
        filters_ : list of W per band (each W shape n_channels x n_components)
        lda_ : trained sklearn LDA
    """
    def __init__(self, fs=250, bands=None, n_components=4, csp_order=4, filter_order=4):
        self.fs = fs
        self.bands = bands or [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
        self.n_components = n_components  # per band
        self.filter_order = filter_order
        # sanity: require even n_components
        if self.n_components % 2 != 0:
            raise ValueError("n_components must be even (pairs).")

    def fit(self, X, y):
        """
        X: (n_trials, n_channels, n_samples)
        y: (n_trials,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("FBCSP_LDA currently supports binary classification only.")
        n_trials, n_ch, n_s = X.shape
        self.filters_ = []
        feat_list = []
        # for each band: preprocess, compute CSP, extract features
        for (low, high) in self.bands:
            Xf = np.zeros_like(X)
            for i in range(n_trials):
                Xf[i] = bandpass_filter(X[i], low, high, self.fs, order=self.filter_order)
            W = csp_fit_binary(Xf, y, n_components=self.n_components)
            self.filters_.append(W)
            feats = extract_logvar_features(Xf, W)  # (n_trials, n_components)
            feat_list.append(feats)
        # concatenate features from all bands
        X_features = np.concatenate(feat_list, axis=1)  # (n_trials, n_bands * n_components)
        self.lda_ = LinearDiscriminantAnalysis()
        self.lda_.fit(X_features, y)
        self.n_features_ = X_features.shape[1]
        return self

    def _transform_X(self, X):
        X = np.asarray(X)
        n_trials = X.shape[0]
        feat_list = []
        for idx, (low, high) in enumerate(self.bands):
            Xf = np.zeros_like(X)
            for i in range(n_trials):
                Xf[i] = bandpass_filter(X[i], low, high, self.fs, order=self.filter_order)
            W = self.filters_[idx]
            feats = extract_logvar_features(Xf, W)
            feat_list.append(feats)
        X_features = np.concatenate(feat_list, axis=1)
        return X_features

    def predict(self, X):
        Xf = self._transform_X(X)
        return self.lda_.predict(Xf)

    def predict_proba(self, X):
        Xf = self._transform_X(X)
        return self.lda_.predict_proba(Xf)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def cross_val_score(self, X, y, cv=5, random_state=42):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            self_cv = FBCSP_LDA(fs=self.fs, bands=self.bands, n_components=self.n_components, filter_order=self.filter_order)
            self_cv.fit(X[train_idx], y[train_idx])
            s = self_cv.score(X[test_idx], y[test_idx])
            scores.append(s)
        return np.array(scores)
