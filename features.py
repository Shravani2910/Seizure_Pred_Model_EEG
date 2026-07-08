"""
Extract hand-crafted features from windowed EEG for the baseline (RF/XGBoost) model.

Per channel: band power (delta, theta, alpha, beta, gamma) + Hjorth parameters
(activity, mobility, complexity) + basic stats (mean, std, skew, kurtosis).
"""
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


def band_power(signal, sfreq):
    freqs, psd = welch(signal, sfreq, nperseg=min(256, len(signal)))
    powers = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs <= hi)
        integrate = getattr(np, "trapezoid", None) or np.trapz  # numpy >=2.0 renamed trapz -> trapezoid
        powers.append(integrate(psd[mask], freqs[mask]))
    return powers


def hjorth_params(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(first_deriv) / (activity + 1e-8))
    complexity = np.sqrt(np.var(second_deriv) / (np.var(first_deriv) + 1e-8)) / (mobility + 1e-8)
    return activity, mobility, complexity


def extract_features_single_window(window, sfreq):
    """window: (n_channels, n_samples) -> 1D feature vector"""
    feats = []
    for ch in range(window.shape[0]):
        sig = window[ch]
        feats.extend(band_power(sig, sfreq))
        feats.extend(hjorth_params(sig))
        feats.extend([np.mean(sig), np.std(sig), skew(sig), kurtosis(sig)])
    return np.array(feats, dtype=np.float32)


def extract_features(X, sfreq):
    """X: (n_windows, n_channels, n_samples) -> (n_windows, n_features)"""
    return np.stack([extract_features_single_window(w, sfreq) for w in X])
