"""
Preprocessing script.
This applies a bandpass filter to remove noise from the raw EEG signals
and then normalizes them using standard z-scores.
"""

import numpy as np
from scipy import signal as sig

from config.settings import SAMPLING_RATE


def preprocess_signals(X_raw):
    """
    Apply bandpass filtering and z-score normalization to all signals.

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_signals, n_samples)
        Raw EEG data straight from the text files.

    Returns
    -------
    X_clean : np.ndarray, same shape as input
        Filtered and normalized signals, ready for feature extraction.
    """
    X_clean = np.copy(X_raw)

    # Design a 4th-order Butterworth bandpass filter
    nyquist = SAMPLING_RATE / 2.0
    low = 0.5 / nyquist
    high = min(85.0 / nyquist, 0.99)   # cap near Nyquist to keep filter stable
    b, a = sig.butter(4, [low, high], btype="band")

    for i in range(X_clean.shape[0]):
        # Zero-phase filtering — applies the filter forward and backward
        # so there's no time delay introduced
        X_clean[i] = sig.filtfilt(b, a, X_clean[i])

        # Z-score: subtract mean, divide by standard deviation
        mean = np.mean(X_clean[i])
        std = np.std(X_clean[i])
        if std > 0:
            X_clean[i] = (X_clean[i] - mean) / std

    return X_clean
