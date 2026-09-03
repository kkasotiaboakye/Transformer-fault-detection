"""
Track 1 -- LSTM Autoencoder anomaly detection (healthy vs fault gate).

NOTE: run_track1() expects an ALREADY PREPROCESSED signal (bandpass
filtered + standardized via preprocess_signal.py), not a raw signal.
If you're using run_pipeline.py's diagnose_signal(), this is handled
for you automatically.

Usage:
    from preprocess_signal import preprocess_signal
    from run_track1 import load_track1, run_track1

    track1_state = load_track1()
    clean_signal = preprocess_signal(raw_signal)
    is_fault, error_value = run_track1(clean_signal, track1_state)

Requires: numpy, scipy, PyWavelets, joblib, tflite-runtime (or tensorflow)
    pip install numpy scipy PyWavelets joblib
    pip install tflite-runtime      # preferred, lightweight
    # OR: pip install tensorflow    # heavier, only if tflite-runtime unavailable

Files required in this folder:
    track1_k20_v3_native.tflite
    track1_temporal_scaler_v3.pkl
    ranked_indices_temporal_v3.npy
    threshold.json
"""

import json
import numpy as np
import joblib
import os
import time


from feature_extraction_track1 import prepare_track1_input

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

def load_track1(model_path='track1_k20_v3_native_fixed.tflite',
                 scaler_path='track1_temporal_scaler_v3.pkl',
                 indices_path='ranked_indices_temporal_v3.npy',
                 threshold_path='threshold.json'):
    """Load everything Track 1 needs once, at startup."""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    scaler = joblib.load(scaler_path)
    ranked_indices = np.load(indices_path)
    top_k_indices = ranked_indices[:20]
    print(f"Track 1 model loaded: {model_path} ({os.path.getsize(model_path)} bytes)")
    with open(threshold_path) as f:
        threshold = json.load(f)['track1_roc_optimal_threshold']

    return {
        'interpreter': interpreter,
        'input_details': interpreter.get_input_details(),
        'output_details': interpreter.get_output_details(),
        'scaler': scaler,
        'top_k_indices': top_k_indices,
        'threshold': threshold,
    }



def run_track1(preprocessed_signal, track1_state):
    t0 = time.time()
    x = prepare_track1_input(
        preprocessed_signal,
        track1_state['scaler'],
        track1_state['top_k_indices'],
    )
    t_features = (time.time() - t0) * 1000

    interpreter = track1_state['interpreter']
    input_details = track1_state['input_details']
    output_details = track1_state['output_details']

    t0 = time.time()
    interpreter.resize_tensor_input(input_details[0]['index'], x.shape)
    interpreter.allocate_tensors()
    t_alloc = (time.time() - t0) * 1000

    t0 = time.time()
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    x_reconstructed = interpreter.get_tensor(output_details[0]['index'])
    t_invoke = (time.time() - t0) * 1000

    print(f"    [track1] features={t_features:.2f}ms  resize+alloc={t_alloc:.2f}ms  invoke={t_invoke:.2f}ms")

    error = float(np.mean((x - x_reconstructed) ** 2))
    is_fault = error > track1_state['threshold']

    return is_fault, error


if __name__ == '__main__':
    # Smoke-test only -- replace with a real incoming signal window.
    # NOTE: this example skips preprocessing for simplicity. In real use,
    # always run preprocess_signal.py on the raw signal first.
    state = load_track1()
    example_signal = np.random.randn(60000)  # placeholder, NOT real data
    fault_detected, err = run_track1(example_signal, state)
    print(f"Fault detected: {fault_detected} (reconstruction error = {err:.4f})")
