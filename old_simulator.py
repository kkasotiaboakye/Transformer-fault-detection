"""
old_simulator.py
================

Signal generation for the notebook (V1) simulator.

This module produces single AE signals for four classes:
    - partial_discharge
    - mechanical
    - arcing
    - healthy

Each faulty class uses a different pulse pattern and frequency range.
A shared propagation-and-noise stage is applied to all signals for
realism, and returns both the pre-noise ("bandpass") signal and the
final noisy signal at the sensor output.

This file has NO dependency on wavelets, features, or ML libraries.
It is deliberately kept as a pure signal-generation module.

Reference:
    Notebook simulator by the user, wrapped here for re-use.
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS = 1_000_000                            # sampling rate (Hz)
T = 0.06                                  # signal duration (s)
N_SAMPLES_PER_SIGNAL = int(FS * T)        # 60000

# Per-fault-class synthesis parameters
FAULT_TYPES = {
    'partial_discharge': {
        'freq_range':    (40e3, 110e3),
        'pulse_pattern': 'phase_dependent',
        'decay_range':   (5e3, 15e3),
    },
    'mechanical': {
        'freq_range':    (150e3, 400e3),
        'pulse_pattern': 'random',
        'decay_range':   (2e3, 8e3),
    },
    'arcing': {
        'freq_range':    (20e3, 90e3),
        'pulse_pattern': 'bursts',
        'decay_range':   (6e3, 12e3),
    },
}

# Canonical class ordering used everywhere downstream
CLASS_ORDER = ['healthy', 'partial_discharge', 'mechanical', 'arcing']


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------

def make_time_axis():
    """Return the shared 60 ms time array at 1 MHz sampling."""
    return np.linspace(0, T, N_SAMPLES_PER_SIGNAL, endpoint=False)


# ---------------------------------------------------------------------------
# Shared propagation + noise stage
# ---------------------------------------------------------------------------

def _apply_propagation_and_noise(signal, t, rng, add_signal_snr):
    """
    Apply propagation bandpass, sensor lowpass, hum, EMI, and white noise.

    Parameters
    ----------
    signal : np.ndarray
        Source AE signal (before propagation).
    t : np.ndarray
        Time axis (from make_time_axis()).
    rng : np.random.Generator
    add_signal_snr : bool
        If True (faulty signal), noise power is computed relative to
        the measured signal power. If False (healthy signal), a fixed
        low noise floor is used.

    Returns
    -------
    bandpass : np.ndarray
        The propagated + sensor-filtered signal BEFORE noise/hum/EMI
        are added. Used for envelope feature extraction so the envelope
        shape isn't contaminated by broadband noise.
    final : np.ndarray
        The full sensor-output signal (bandpass + hum + EMI + white).
        Used for wavelet feature extraction (matches what a real sensor
        would deliver).
    """
    lowcut = rng.uniform(20e3, 50e3)
    highcut = rng.uniform(150e3, 300e3)
    b, a = butter(2, [lowcut, highcut], btype='band', fs=FS)
    propagated = filtfilt(b, a, signal) * rng.uniform(0.6, 1.0)

    sensor_cutoff = rng.uniform(200e3, 350e3)
    b_s, a_s = butter(2, sensor_cutoff, btype='low', fs=FS)
    measured = filtfilt(b_s, a_s, propagated) * rng.uniform(0.7, 1.3)

    load = rng.uniform(0.3, 1.0)
    emi_active = rng.choice([0, 1], p=[0.3, 0.7])
    emi = (emi_active * rng.uniform(0.001, 0.05) * load
           * np.sin(2 * np.pi * rng.uniform(10e3, 50e3) * t))
    hum = 0.001 * load * np.sin(2 * np.pi * 60 * t)
    for h in (2, 3, 5):
        hum += (0.001 * load / h) * np.sin(2 * np.pi * 60 * h * t)

    snr_db = rng.uniform(0, 10)
    if add_signal_snr:
        signal_power = float(np.mean(measured ** 2))
        noise_power = signal_power / (10 ** (snr_db / 10)) if signal_power > 0 \
            else 1e-9
    else:
        noise_power = 1e-6 / (10 ** (snr_db / 10))
    white = np.sqrt(noise_power) * rng.standard_normal(len(measured))

    final = (measured + hum + white + emi).astype(np.float32)
    bandpass = measured.astype(np.float32)
    return bandpass, final


# ---------------------------------------------------------------------------
# Faulty signal generator
# ---------------------------------------------------------------------------

def generate_one_faulty_signal(t, fault_type, rng):
    """
    Generate a single faulty signal (60 ms).

    Returns (bandpass_signal, final_signal) tuple.
    """
    params = FAULT_TYPES[fault_type]

    severity = rng.choice([1, 2, 3])
    severity_map = {1: (3, 6), 2: (8, 15), 3: (20, 40)}
    n_pulses = rng.integers(*severity_map[severity])

    pulse_times = []

    if params['pulse_pattern'] == 'phase_dependent':
        ac_period = 1.0 / 60.0
        n_cycles = int(T / ac_period) + 1
        for cycle in range(n_cycles):
            for _ in range(max(1, n_pulses // n_cycles)):
                phase = rng.uniform(0.7, 1.0)
                t0 = cycle * ac_period + phase * ac_period
                if 0.001 < t0 < T - 0.001:
                    pulse_times.append(t0)

    elif params['pulse_pattern'] == 'random':
        for _ in range(n_pulses):
            r = rng.random()
            t0 = 0.0005 + (T - 0.001 - 0.0005) * (r ** 0.5)
            pulse_times.append(t0)
        pulse_times = sorted(pulse_times)

    elif params['pulse_pattern'] == 'bursts':
        n_clusters = max(2, n_pulses // 4)
        centres = rng.uniform(0.001, T - 0.002, n_clusters)
        for center in centres:
            for _ in range(n_pulses // n_clusters):
                t0 = center + rng.normal(0, 0.0003)
                if 0.0005 < t0 < T - 0.001:
                    pulse_times.append(t0)

    pulse_times = sorted(pulse_times)[:n_pulses]

    signal = np.zeros(N_SAMPLES_PER_SIGNAL, dtype=np.float32)
    freq_min, freq_max = params['freq_range']
    alpha_min, alpha_max = params['decay_range']

    for t0 in pulse_times:
        A = rng.uniform(0.5, 1.5) * severity
        f0 = rng.uniform(freq_min, freq_max)
        alpha = rng.uniform(alpha_min, alpha_max)
        # Compute the pulse only for samples where t >= t0 to avoid
        # exponent overflow at t << t0 (where -alpha * (t - t0) is a
        # large positive number). The mask assignment below relies on
        # np.exp receiving only non-positive arguments.
        mask = t >= t0
        dt = t[mask] - t0
        pulse = np.zeros_like(t)
        pulse[mask] = A * np.exp(-alpha * dt) * np.sin(2 * np.pi * f0 * dt)
        signal += pulse.astype(np.float32)

    return _apply_propagation_and_noise(signal, t, rng, add_signal_snr=True)


# ---------------------------------------------------------------------------
# Healthy signal generator
# ---------------------------------------------------------------------------

def generate_one_healthy_signal(t, rng):
    """
    Generate a single healthy signal (60 ms, noise only, no AE source).

    Returns (bandpass_signal, final_signal) tuple.
    """
    signal = np.zeros(N_SAMPLES_PER_SIGNAL, dtype=np.float32)
    return _apply_propagation_and_noise(signal, t, rng, add_signal_snr=False)


# ---------------------------------------------------------------------------
# Simple self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    t = make_time_axis()

    for cls in CLASS_ORDER:
        if cls == 'healthy':
            bp, fin = generate_one_healthy_signal(t, rng)
        else:
            bp, fin = generate_one_faulty_signal(t, cls, rng)
        print(f'{cls:22s}  bandpass RMS = {float(np.sqrt(np.mean(bp**2))):.4f}  '
              f'final RMS = {float(np.sqrt(np.mean(fin**2))):.4f}')
