"""
Full pipeline: preprocessing -> Track 1 (anomaly gate) -> Track 2 (fault classification).

Logic:
    1. Preprocess the raw signal ONCE (bandpass filter + standardize).
    2. Run Track 1 on the preprocessed signal.
    3. If Track 1 says HEALTHY -> report healthy, stop. Track 2 does not run.
    4. If Track 1 says FAULT -> run Track 2 on the SAME preprocessed signal
       to identify which fault type it is.

IMPORTANT: Track 1 and Track 2 were both trained on signals that went
through the SAME preprocessing (bandpass 20-400 kHz + z-score standardize).
Preprocessing happens ONCE, here, before either track runs -- do not
preprocess again inside either track, and do not feed a raw/unpreprocessed
signal directly to either track's model.

Usage:
    python run_pipeline.py

Folder layout expected (see README.md for the full file list):
    preprocess_signal.py -- shared preprocessing, used by both tracks
    track1/  -- Track 1 model + feature extraction + this track's files
    track2/  -- Track 2 model + feature extraction + this track's files
"""

import sys
import numpy as np

sys.path.insert(0, 'track1')
sys.path.insert(0, 'track2')

from preprocess_signal import preprocess_signal
from track1.run_track1 import load_track1, run_track1
from track2.run_track2 import load_track2, classify_track2


def load_pipeline():
    """Load both tracks once, at startup. Call this before diagnose_signal()."""
    track1_state = load_track1(
        model_path='track1/track1_k20_v3_native_fixed.tflite',
        scaler_path='track1/track1_temporal_scaler_v3.pkl',
        indices_path='track1/ranked_indices_temporal_v3.npy',
        threshold_path='track1/threshold.json',
    )
    track2_model, track2_labels = load_track2(
        model_path='track2/xgb_production_model.json',
        label_map_path='track2/label_mapping.json',
    )
    return {
        'track1': track1_state,
        'track2_model': track2_model,
        'track2_labels': track2_labels,
    }


def diagnose_signal(raw_signal, pipeline_state, fs=1_000_000):
    """
    raw_signal: 1D numpy array, ONE raw AE signal window, straight from
                the sensor, untouched. Preprocessing happens inside this
                function -- do not preprocess it yourself first.
    pipeline_state: dict returned by load_pipeline().

    Returns a dict, always containing 'status' and 'track1_error':
        {'status': 'healthy', 'track1_error': 0.41}
        {'status': 'fault', 'track1_error': 812.3,
         'fault_type': 'mechanical', 'class_probabilities': {...}}
    """
    clean_signal = preprocess_signal(raw_signal, fs=fs)

    is_fault, track1_error = run_track1(clean_signal, pipeline_state['track1'])

    if not is_fault:
        return {
            'status': 'healthy',
            'track1_error': track1_error,
        }

    fault_type, class_probs = classify_track2(
        clean_signal,
        pipeline_state['track2_model'],
        pipeline_state['track2_labels'],
        fs=fs,
    )

    return {
        'status': 'fault',
        'track1_error': track1_error,
        'fault_type': fault_type,
        'class_probabilities': class_probs,
    }


if __name__ == '__main__':
    print("Loading Track 1 and Track 2 models...")
    pipeline = load_pipeline()
    print("Ready.\n")

    # Smoke-test only -- replace with a real incoming signal window
    # (60,000 samples at 1 MHz, matching the training data's window length).
    example_signal = np.random.randn(60000)

    result = diagnose_signal(example_signal, pipeline)

    if result['status'] == 'healthy':
        print(f"Result: HEALTHY (Track 1 reconstruction error = {result['track1_error']:.4f})")
    else:
        print(f"Result: FAULT DETECTED (Track 1 reconstruction error = {result['track1_error']:.4f})")
        print(f"  Fault type: {result['fault_type']}")
        print(f"  Class probabilities: {result['class_probabilities']}")
