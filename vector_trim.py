import numpy as np
from old_simulator import make_time_axis, generate_one_healthy_signal, generate_one_faulty_signal, CLASS_ORDER
from preprocess_signal import preprocess_signal
from track1.feature_extraction_track1 import extract_features_temporal, extract_features_temporal_vectorized

t = make_time_axis()
rng = np.random.default_rng(0)
fault_types = [c for c in CLASS_ORDER if c != 'healthy']

max_diffs = []
for i in range(20):
    if i % 4 == 0:
        _, sig = generate_one_faulty_signal(t, fault_types[i % len(fault_types)], rng)
    else:
        _, sig = generate_one_healthy_signal(t, rng)
    clean = preprocess_signal(sig)

    old_out = extract_features_temporal(clean)
    new_out = extract_features_temporal_vectorized(clean)

    diff = np.abs(old_out - new_out).max()
    max_diffs.append(diff)
    print(f"sample {i}: max abs diff = {diff:.6f}")

print(f"\nOverall max diff across all samples: {max(max_diffs):.6f}")