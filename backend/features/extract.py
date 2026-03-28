"""
Feature extraction script.
This splits each EEG signal into pieces and extracts wavelets, 
frequencies, and entropy features to feed into the model.
"""

import math
import time
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq

from config.settings import SAMPLING_RATE, DWT_LEVEL, N_SEGMENTS


# ── DB4 wavelet filters (no PyWavelets dependency) ───────

def _db4_filters():
    """Got these db4 filter coefficients for the wavelets"""
    h = np.array([
        -0.010597401784997278,  0.032883011666982945,
         0.030841381835986965, -0.18703481171888114,
        -0.027983769416983849,  0.63088076792959036,
         0.71484657055254153,   0.23037781330885523,
    ])
    g = np.array([(-1)**k * h[len(h) - 1 - k] for k in range(len(h))])
    return h, g


def _dwt_decompose(signal_data):
    """Do the wavelet transform manually using numpy convolution"""
    h, g = _db4_filters()
    coefficients = []
    approx = signal_data.copy()

    for _ in range(DWT_LEVEL):
        detail = np.convolve(approx, g, mode="full")[::2]
        approx = np.convolve(approx, h, mode="full")[::2]
        coefficients.insert(0, detail)

    coefficients.insert(0, approx)
    return coefficients


# ── Per-domain feature functions ─────────────────────────

def _dwt_features(coefficients):
    """15 statistics per wavelet sub-band. 5 bands x 15 = 75 features."""
    features = []
    for c in coefficients:
        if len(c) == 0:
            features.extend([0.0] * 15)
            continue

        abs_c = np.abs(c)
        c_sq = c ** 2
        energy = np.sum(c_sq) + 1e-12
        normalized = c_sq / energy
        mean_val = np.mean(c)
        std_val = np.std(c)

        features.extend([
            mean_val,
            np.max(c),
            np.min(c),
            np.var(c),
            -np.sum(normalized * np.log2(normalized + 1e-12)),  # wavelet entropy
            np.sum(c_sq),                                       # energy
            np.max(abs_c),
            np.min(abs_c),
            std_val,
            float(kurtosis(c, fisher=True, bias=False)),
            float(skew(c, bias=False)),
            np.sqrt(np.mean(c_sq)),                             # RMS
            np.sum(np.diff(np.sign(c)) != 0) / len(c),         # zero-crossing rate
            np.mean(np.abs(c - mean_val)),                      # mean absolute deviation
            std_val / (np.abs(mean_val) + 1e-12),               # coefficient of variation
        ])

    return features


def _time_features(segment):
    """
    17 time-domain features.
    Note: Hjorth Activity was dropped (identical to Variance),
    and Total Variation was dropped (identical to Line Length).
    """
    n = len(segment)
    d1 = np.diff(segment)
    d2 = np.diff(d1)

    var0 = np.var(segment)
    var1 = np.var(d1)
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt(np.var(d2) / (var1 + 1e-12)) / (mobility + 1e-12)

    teager = 0.0
    if n >= 3:
        teager = np.mean(np.abs(segment[1:-1] ** 2 - segment[:-2] * segment[2:]))

    return [
        np.mean(segment),
        np.std(segment),
        var0,
        np.max(segment),
        np.min(segment),
        np.ptp(segment),                                            # peak-to-peak
        float(kurtosis(segment, fisher=True, bias=False)),
        float(skew(segment, bias=False)),
        np.sqrt(np.mean(segment ** 2)),                             # RMS
        np.sum(np.diff(np.sign(segment)) != 0) / n,                # zero-crossing rate
        np.mean(np.abs(segment)),                                   # mean absolute value
        mobility,                                                   # Hjorth mobility
        complexity,                                                 # Hjorth complexity
        np.sum(np.abs(d1)),                                         # line length
        teager,
        np.percentile(segment, 75) - np.percentile(segment, 25),   # IQR
        np.median(np.abs(segment - np.median(segment))),            # median abs deviation
    ]


def _frequency_features(segment, fs=SAMPLING_RATE):
    """22 frequency-domain features from FFT power spectrum."""
    n = len(segment)
    yf = fft(segment)
    xf = fftfreq(n, 1.0 / fs)

    pos_mask = xf > 0
    freqs = xf[pos_mask]
    psd = np.abs(yf[pos_mask]) ** 2

    total_power = np.sum(psd) + 1e-12
    psd_norm = psd / total_power
    cumulative = np.cumsum(psd_norm)

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 85),
    }

    band_power = {}
    band_feats = []
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp = np.sum(psd[mask])
        band_power[name] = bp
        band_feats += [bp, bp / total_power]

    idx_95 = min(np.searchsorted(cumulative, 0.95), len(freqs) - 1)
    idx_85 = min(np.searchsorted(cumulative, 0.85), len(freqs) - 1)
    idx_50 = min(np.searchsorted(cumulative, 0.50), len(freqs) - 1)

    centroid = np.sum(freqs * psd) / total_power

    return [
        total_power,
        freqs[np.argmax(psd)],                                      # peak frequency
        np.sum(freqs * psd_norm),                                   # mean frequency
        freqs[idx_50],                                              # median frequency
        -np.sum(psd_norm * np.log2(psd_norm + 1e-12)),              # spectral entropy
        *band_feats,                                                # 10 band features
        band_power["theta"] / (band_power["alpha"] + 1e-12),
        band_power["delta"] / (band_power["beta"] + 1e-12),
        (band_power["delta"] + band_power["theta"]) /
            (band_power["alpha"] + band_power["beta"] + 1e-12),
        freqs[idx_95],                                              # spectral edge 95%
        freqs[idx_85],                                              # spectral edge 85%
        centroid,
        np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total_power),
    ]


# ── Non-linear features ─────────────────────────────────

def _sample_entropy(x, m=2, r_factor=0.2):
    """Sample entropy — capped at 100 samples for speed."""
    n = min(len(x), 100)
    x = x[:n]
    r = r_factor * np.std(x)

    if r < 1e-10 or n < m + 2:
        return 0.0

    def _count_matches(template_len):
        templates = np.array([x[i:i + template_len] for i in range(n - template_len)])
        count = 0
        for i in range(len(templates)):
            count += np.sum(np.max(np.abs(templates - templates[i]), axis=1) < r) - 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    return -np.log((A + 1e-12) / (B + 1e-12)) if B > 0 else 0.0


def _permutation_entropy(x, order=3, delay=1):
    """Permutation entropy — complexity via ordinal patterns."""
    n = len(x)
    if n < order * delay + 1:
        return 0.0

    n_perms = math.factorial(order)
    indices = np.arange(order) * delay
    n_vectors = n - (order - 1) * delay

    patterns = [tuple(np.argsort(x[i + indices])) for i in range(n_vectors)]
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / n_vectors

    return -np.sum(probs * np.log2(probs + 1e-12)) / np.log2(n_perms)


def _higuchi_fd(x, kmax=8):
    """Higuchi fractal dimension — roughness / complexity of the signal."""
    n = len(x)
    ks = np.arange(1, kmax + 1)
    lengths = []

    for k in ks:
        Lk = []
        for m_start in range(1, k + 1):
            idx = np.arange(0, int(np.floor((n - m_start) / k)), dtype=int)
            if len(idx) < 2:
                continue
            s = x[m_start + idx * k]
            Lk.append(np.sum(np.abs(np.diff(s))) * (n - 1) / (k * len(idx) * k + 1e-12))
        lengths.append(np.mean(Lk) if Lk else 1e-12)

    lengths = np.array(lengths)
    valid = lengths > 0
    if np.sum(valid) < 3:
        return 1.0

    try:
        slope, _ = np.polyfit(np.log(1.0 / ks[valid]), np.log(lengths[valid]), 1)
    except Exception:
        slope = 1.0

    return slope


def _dfa(x):
    """
    Detrended Fluctuation Analysis — measures long-range correlations.
    Returns the scaling exponent alpha.
    """
    N = len(x)
    if N < 16:
        return 0.5

    y = np.cumsum(x - np.mean(x))

    nvals = np.unique(np.logspace(np.log10(4), np.log10(N // 4), 20).astype(int))
    nvals = nvals[nvals >= 4]

    if len(nvals) < 3:
        return 0.5

    fluctuations = []
    for n_win in nvals:
        n_windows = N // n_win
        if n_windows < 1:
            continue

        F_n = 0.0
        for w in range(n_windows):
            segment = y[w * n_win:(w + 1) * n_win]
            x_axis = np.arange(n_win)
            coeffs = np.polyfit(x_axis, segment, 1)
            trend = np.polyval(coeffs, x_axis)
            F_n += np.sum((segment - trend) ** 2)

        F_n = np.sqrt(F_n / (n_windows * n_win))
        if F_n > 0:
            fluctuations.append((n_win, F_n))

    if len(fluctuations) < 3:
        return 0.5

    ns, Fs = zip(*fluctuations)
    try:
        slope, _ = np.polyfit(np.log(np.array(ns)), np.log(np.array(Fs)), 1)
    except Exception:
        slope = 0.5

    return float(slope)


def _nonlinear_features(segment):
    """6 non-linear complexity features."""
    cumsum = np.cumsum(segment - np.mean(segment))

    return [
        _sample_entropy(segment),
        _permutation_entropy(segment, 3, 1),
        _permutation_entropy(segment, 4, 1),
        _dfa(segment),                                              # real DFA
        _higuchi_fd(segment, 8),
        np.std(cumsum) / (len(segment) ** 0.5 + 1e-12),            # Hurst-like
    ]


# ── Public API ───────────────────────────────────────────

N_PER_SEGMENT = 75 + 17 + 22 + 6   # = 120
N_FEATURES = N_PER_SEGMENT * 6     # = 720


def extract_features_single(signal_data):
    """
    Extract 720 features from one EEG signal.

    The signal is split into 16 equal segments. Per-segment features
    are aggregated with 6 statistics (mean, std, max, min, median, IQR).
    """
    segment_len = len(signal_data) // N_SEGMENTS
    segment_features = []

    for i in range(N_SEGMENTS):
        seg = signal_data[i * segment_len : (i + 1) * segment_len]

        dwt_coeffs = _dwt_decompose(seg)
        feat = (
            _dwt_features(dwt_coeffs)
            + _time_features(seg)
            + _frequency_features(seg)
            + _nonlinear_features(seg)
        )
        segment_features.append(np.array(feat, dtype=np.float64))

    segment_features = np.array(segment_features)   # shape: (16, 120)

    iqr = (
        np.percentile(segment_features, 75, axis=0)
        - np.percentile(segment_features, 25, axis=0)
    )

    aggregated = np.concatenate([
        segment_features.mean(axis=0),
        segment_features.std(axis=0),
        segment_features.max(axis=0),
        segment_features.min(axis=0),
        np.median(segment_features, axis=0),
        iqr,
    ])

    return aggregated


def get_feature_names():
    """
    Build human-readable names for all 720 features.
    Structure: 120 per-segment features x 6 aggregation statistics.
    """
    dwt_bands = [
        "A4 (delta 0-5Hz)",
        "D4 (theta 5-11Hz)",
        "D3 (alpha 11-22Hz)",
        "D2 (beta 22-43Hz)",
        "D1 (gamma 43-87Hz)",
    ]
    dwt_stats = [
        "Mean", "Max", "Min", "Variance", "Wavelet Entropy",
        "Energy", "Max Abs", "Min Abs", "Std Dev", "Kurtosis",
        "Skewness", "RMS", "Zero-Cross Rate", "Mean Abs Dev", "Coeff Var",
    ]

    time_names = [
        "Signal Mean", "Signal Std", "Signal Variance",
        "Signal Max", "Signal Min", "Peak-to-Peak Range",
        "Kurtosis", "Skewness", "RMS",
        "Zero-Crossing Rate", "Mean Abs Value",
        "Hjorth Mobility", "Hjorth Complexity",
        "Line Length", "Teager Energy",
        "IQR", "Median Abs Dev",
    ]

    freq_names = [
        "Total Power", "Peak Frequency", "Mean Frequency",
        "Median Frequency", "Spectral Entropy",
        "Delta Power (abs)", "Delta Power (rel)",
        "Theta Power (abs)", "Theta Power (rel)",
        "Alpha Power (abs)", "Alpha Power (rel)",
        "Beta Power (abs)", "Beta Power (rel)",
        "Gamma Power (abs)", "Gamma Power (rel)",
        "Theta/Alpha Ratio", "Delta/Beta Ratio", "Slow/Fast Ratio",
        "Spectral Edge 95%", "Spectral Edge 85%",
        "Spectral Centroid", "Spectral Spread",
    ]

    nonlinear_names = [
        "Sample Entropy", "Perm Entropy (ord 3)",
        "Perm Entropy (ord 4)", "DFA",
        "Higuchi Fractal Dim", "Hurst Exponent",
    ]

    segment_names = []
    for band in dwt_bands:
        for stat in dwt_stats:
            segment_names.append(f"DWT {band} {stat}")
    segment_names.extend(time_names)
    segment_names.extend(freq_names)
    segment_names.extend(nonlinear_names)

    agg_labels = ["Mean", "Std", "Max", "Min", "Median", "IQR"]

    all_names = []
    for agg in agg_labels:
        for seg_name in segment_names:
            all_names.append(f"{agg} | {seg_name}")

    return all_names


def extract_features_batch(X_signals):
    """Extract features from multiple signals with progress updates."""
    print(f"  Extracting features from {X_signals.shape[0]} signals...")
    t0 = time.time()

    features = []
    for i in range(X_signals.shape[0]):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{X_signals.shape[0]} done...")
        features.append(extract_features_single(X_signals[i]))

    X_features = np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)
    elapsed = time.time() - t0
    print(f"  Done: {X_features.shape[1]} features x {X_features.shape[0]} signals in {elapsed:.1f}s")
    print(f"  Feature matrix shape: {X_features.shape}")

    return X_features
