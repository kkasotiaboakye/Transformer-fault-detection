"""
Feature extraction for Track 1 (LSTM Autoencoder anomaly detection).

Converts a PREPROCESSED AE signal window into the (32, 30) temporal feature
tensor the Track 1 model expects. Run preprocess_signal.py (shared with
Track 2) on the raw signal FIRST, then pass the result here.

Pipeline: raw signal -> [preprocess_signal.py: bandpass + standardize]
          -> [this file: DWT temporal features]
Requires: numpy, scipy, PyWavelets
    pip install numpy scipy PyWavelets
"""

import numpy as np
import pywt
from scipy.stats import entropy, kurtosis
import time

WAVELET = 'db4'
DWT_LEVEL = 5
N_SEGMENTS = 32       # timesteps fed to the LSTM

def fast_entropy(x):
    # Equivalent to scipy.stats.entropy(x) when x is strictly positive
    # (guaranteed here since input is np.abs(seg) + 1e-10)
    p = x / np.sum(x)
    return -np.sum(p * np.log(p))

def fast_kurtosis(x):
    # Equivalent to scipy.stats.kurtosis(x) -- Fisher's definition,
    # biased estimator (scipy's defaults: fisher=True, bias=True)
    x = x - np.mean(x)
    m2 = np.mean(x ** 2)
    if m2 == 0:
        return 0.0
    m4 = np.mean(x ** 4)
    return m4 / (m2 ** 2) - 3.0


def extract_features_temporal(window, n_segments=N_SEGMENTS):
    t0 = time.time()
    coeffs = pywt.wavedec(window, wavelet=WAVELET, level=DWT_LEVEL)
    t_wavedec = (time.time() - t0) * 1000

    t0 = time.time()
    split_subbands = [np.array_split(sb, n_segments) for sb in coeffs]
    t_split = (time.time() - t0) * 1000

    t0 = time.time()
    timesteps = np.zeros((n_segments, 30), dtype=np.float32)
    for segment_idx in range(n_segments):
        col = 0
        for subband_chunks in split_subbands:
            seg = subband_chunks[segment_idx]
            timesteps[segment_idx, col]     = np.mean(seg)
            timesteps[segment_idx, col + 1] = np.std(seg)
            timesteps[segment_idx, col + 2] = np.sum(seg ** 2)
            timesteps[segment_idx, col + 3] = fast_entropy(np.abs(seg) + 1e-10)
            timesteps[segment_idx, col + 4] = fast_kurtosis(seg)
            col += 5
    t_stats_loop = (time.time() - t0) * 1000

    print(f"      [features] wavedec={t_wavedec:.2f}ms  split={t_split:.2f}ms  stats_loop={t_stats_loop:.2f}ms")

    return timesteps

def _vectorized_stats_for_chunks(chunks_2d):
    """Given a 2D array (n_chunks, chunk_len), compute all 5 stats per row at once."""
    if chunks_2d.shape[0] == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty

    means = np.mean(chunks_2d, axis=1)
    stds = np.std(chunks_2d, axis=1)
    energies = np.sum(chunks_2d ** 2, axis=1)

    p = np.abs(chunks_2d) + 1e-10
    p = p / np.sum(p, axis=1, keepdims=True)
    ents = -np.sum(p * np.log(p), axis=1)

    centered = chunks_2d - np.mean(chunks_2d, axis=1, keepdims=True)
    m2 = np.mean(centered ** 2, axis=1)
    m4 = np.mean(centered ** 4, axis=1)
    kurts = np.where(m2 > 0, m4 / (m2 ** 2) - 3.0, 0.0)

    return means, stds, energies, ents, kurts


def extract_features_temporal_vectorized(window, n_segments=N_SEGMENTS):
    coeffs = pywt.wavedec(window, wavelet=WAVELET, level=DWT_LEVEL)

    timesteps = np.zeros((n_segments, 30), dtype=np.float32)
    col = 0
    for subband in coeffs:
        L = len(subband)
        base = L // n_segments
        remainder = L % n_segments  # how many chunks get one extra sample

        # First `remainder` chunks: (base + 1) samples each -- matches
        # np.array_split's front-loaded remainder distribution exactly.
        head_len = remainder * (base + 1)
        head = subband[:head_len].reshape(remainder, base + 1) if remainder > 0 else np.zeros((0, 0))
        h_mean, h_std, h_energy, h_ent, h_kurt = _vectorized_stats_for_chunks(head)

        # Remaining chunks: exactly `base` samples each.
        tail = subband[head_len:].reshape(n_segments - remainder, base) if base > 0 else np.zeros((n_segments - remainder, 0))
        t_mean, t_std, t_energy, t_ent, t_kurt = _vectorized_stats_for_chunks(tail)

        timesteps[:, col]     = np.concatenate([h_mean, t_mean])
        timesteps[:, col + 1] = np.concatenate([h_std, t_std])
        timesteps[:, col + 2] = np.concatenate([h_energy, t_energy])
        timesteps[:, col + 3] = np.concatenate([h_ent, t_ent])
        timesteps[:, col + 4] = np.concatenate([h_kurt, t_kurt])

        col += 5

    return timesteps
def prepare_track1_input(preprocessed_signal, scaler, top_k_indices):
    """
    Track 1 feature pipeline: preprocessed signal -> model-ready input.

    preprocessed_signal: 1D numpy array, output of preprocess_signal.py
                         (bandpass filtered + standardized). NOT a raw signal.
    scaler: the loaded StandardScaler object, fit on all 30 features
            during training (track1_temporal_scaler_v3.pkl).
    top_k_indices: numpy array of 20 feature-column indices (out of 30)
                   selected by MCFS during training
                   (ranked_indices_temporal_v3.npy[:20]).

    Returns: np.ndarray of shape (1, 32, 20), ready for the TFLite model.
    """
    feats_30 = extract_features_temporal_vectorized(preprocessed_signal)  # (32, 30)

    # Scale using the 30-feature scaler BEFORE selecting columns -- the
    # scaler's mean/std were computed on all 30 features, so scaling must
    # happen first, then column selection. Doing this in the other order
    # silently produces wrong numbers.
    feats_scaled = scaler.transform(feats_30.reshape(-1, 30)).reshape(feats_30.shape)
    feats_20 = feats_scaled[:, top_k_indices]              # (32, 20)

    return np.expand_dims(feats_20, axis=0).astype(np.float32)  # (1, 32, 20)
