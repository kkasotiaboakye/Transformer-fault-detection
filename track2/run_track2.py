"""
Track 2 -- XGBoost multiclass fault classification.

Thin wrapper around feature_extraction.py so the top-level pipeline
(run_pipeline.py) can call Track 1 and Track 2 with a consistent interface.
Only called when Track 1 has already flagged the signal as a fault.

Requires: xgboost, numpy, PyWavelets, scipy
    pip install xgboost numpy PyWavelets scipy

Files required in this folder:
    xgb_production_model.json
    label_mapping.json
"""

import json
import numpy as np
import xgboost as xgb

from track2.feature_extraction import extract_features, features_to_vector


def load_track2(model_path='xgb_production_model.json',
                 label_map_path='label_mapping.json'):
    """Load the Track 2 model and label map once, at startup."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(label_map_path) as f:
        raw_map = json.load(f)
    label_map = {int(k): v for k, v in raw_map.items()}

    return model, label_map


def classify_track2(preprocessed_signal, model, label_map, fs=1_000_000):
    """
    preprocessed_signal: 1D numpy array, output of preprocess_signal.py
                         (bandpass filtered + standardized). This is the
                         SAME preprocessed signal passed to Track 1 --
                         both tracks share identical preprocessing.
    model, label_map: as returned by load_track2().

    Returns: (predicted_label: str, class_probabilities: dict)
    """
    feats = extract_features(preprocessed_signal, fs=fs)
    X = np.array([features_to_vector(feats)])  # shape (1, 9)

    pred_idx = int(model.predict(X)[0])
    pred_probs = model.predict_proba(X)[0]

    predicted_label = label_map[pred_idx]
    prob_dict = {label_map[i]: float(p) for i, p in enumerate(pred_probs)}

    return predicted_label, prob_dict
