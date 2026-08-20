"""
Minimal example: load the production model and classify one AE signal window.

This is a template, not a complete deployment pipeline - it does not cover
how signal windows arrive from the sensor (streaming vs. batch), which is a
deployment decision outside the scope of this handoff.

Requires: xgboost, numpy, PyWavelets, scipy
    pip install xgboost numpy PyWavelets scipy
"""

import json
import numpy as np
import xgboost as xgb

from feature_extraction import extract_features, features_to_vector, FEATURE_NAMES

MODEL_PATH = 'xgb_production_model.json'
LABEL_MAP_PATH = 'label_mapping.json'


def load_model_and_labels():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(LABEL_MAP_PATH) as f:
        raw_map = json.load(f)
    # keys may load as strings; normalize to int -> label
    label_map = {int(k): v for k, v in raw_map.items()}

    return model, label_map


def classify_window(signal_window, model, label_map, fs=1_000_000):
    """
    signal_window: 1D numpy array, one AE signal window at the model's
                   trained sampling rate (fs).
    Returns: (predicted_label:str, class_probabilities:dict)
    """
    feats = extract_features(signal_window, fs=fs)
    X = np.array([features_to_vector(feats)])  # shape (1, 9), correct column order

    pred_idx = int(model.predict(X)[0])
    pred_probs = model.predict_proba(X)[0]

    predicted_label = label_map[pred_idx]
    prob_dict = {label_map[i]: float(p) for i, p in enumerate(pred_probs)}

    return predicted_label, prob_dict


if __name__ == '__main__':
    model, label_map = load_model_and_labels()

    # Example only - replace with a real incoming signal window
    example_window = np.random.randn(20000)  # placeholder, NOT real data

    label, probs = classify_window(example_window, model, label_map)
    print(f"Predicted fault type: {label}")
    print(f"Class probabilities: {probs}")
