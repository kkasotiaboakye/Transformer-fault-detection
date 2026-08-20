import os
import socket
import threading
import queue
import struct
import time
from collections import deque
import numpy as np
from run_pipeline import load_pipeline, diagnose_signal

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 65432
SAMPLE_RATE = 1_000_000
DTYPE = np.float32
BATCH_MS = 60
BATCH_SAMPLES = int(SAMPLE_RATE * BATCH_MS / 1000)

HEADER_FORMAT = ">QIB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

CLASS_ORDER = ['healthy', 'partial_discharge', 'mechanical', 'arcing']

# --- Downsampling for the UI waveform ---
# Full rate is 1 MHz; UI can't render that. We downsample by taking every Nth sample.
UI_DOWNSAMPLE_FACTOR = 2000  # 1 MHz -> 500 samples per second for UI
UI_SAMPLES_PER_BATCH = BATCH_SAMPLES // UI_DOWNSAMPLE_FACTOR  # 30 per batch

# --- Shared state for web server (imported by web_server.py) ---
ui_queue = queue.Queue()   # thread-safe: web server reads from this
stats = {
    'total_batches': 0,
    'faults_detected': 0,
    'total_latency_ms': 0,
    'start_time': None,
}

data_queue = queue.Queue()
stop_flag = threading.Event()
gap_count = [0]
has_announced_start = [False]


def recv_exact(conn, num_bytes):
    data = b""
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet:
            return None
        data += packet
    return data


def network_thread(conn):
    running_total = 0
    while not stop_flag.is_set():
        header = recv_exact(conn, HEADER_SIZE)
        if header is None:
            stop_flag.set()
            break

        cumulative_count, chunk_byte_len, label_code = struct.unpack(HEADER_FORMAT, header)
        payload = recv_exact(conn, chunk_byte_len)
        if payload is None:
            stop_flag.set()
            break

        if not has_announced_start[0]:
            print("Receiver is now receiving data.")
            has_announced_start[0] = True
            stats['start_time'] = time.time()

        samples = np.frombuffer(payload, dtype=DTYPE)
        expected_total = running_total + len(samples)

        if cumulative_count != expected_total:
            missing = cumulative_count - expected_total
            gap_count[0] += 1
            print("GAP DETECTED: " + str(missing) + " samples missing")
            running_total = cumulative_count
        else:
            running_total = expected_total

        data_queue.put((label_code, samples))


def receiver_main():
    """Runs the inference loop. Called from web_server.py as a background thread."""
    print("Loading inference pipeline...")
    pipeline = load_pipeline()
    print("Pipeline ready.")

    accum_buffer = np.array([], dtype=DTYPE)
    pending_labels = deque()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)
        print("Waiting for sender...")

        conn, addr = server_sock.accept()
        recv_thread = threading.Thread(target=network_thread, args=(conn,), daemon=True)
        recv_thread.start()

        while not stop_flag.is_set():
            try:
                label_code, samples = data_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            accum_buffer = np.concatenate([accum_buffer, samples])
            pending_labels.append([label_code, len(samples)])

            while len(accum_buffer) >= BATCH_SAMPLES:
                batch = accum_buffer[:BATCH_SAMPLES]
                accum_buffer = accum_buffer[BATCH_SAMPLES:]

                remaining = BATCH_SAMPLES
                contributing_labels = set()
                while remaining > 0 and pending_labels:
                    seg_label, seg_count = pending_labels[0]
                    take = min(remaining, seg_count)
                    contributing_labels.add(seg_label)
                    remaining -= take
                    if take == seg_count:
                        pending_labels.popleft()
                    else:
                        pending_labels[0][1] -= take

                fault_labels = [l for l in contributing_labels if l != 0]
                gt = CLASS_ORDER[fault_labels[0]] if fault_labels else "healthy"

                t0 = time.time()
                result = diagnose_signal(batch, pipeline)
                elapsed_ms = (time.time() - t0) * 1000

                # Update stats
                stats['total_batches'] += 1
                stats['total_latency_ms'] += elapsed_ms
                if result['status'] == 'fault':
                    stats['faults_detected'] += 1

                # Downsample batch for UI waveform (take every Nth sample)
                waveform = batch[::UI_DOWNSAMPLE_FACTOR].tolist()

                # Build a payload for the web server to broadcast
                ui_payload = {
                    'batch': stats['total_batches'],
                    'status': result['status'],
                    'ground_truth': gt,
                    'latency_ms': round(elapsed_ms, 1),
                    'waveform': waveform,
                    'track1_error': round(result.get('track1_error', 0), 4),
                }
                if result['status'] == 'fault':
                    ui_payload['predicted'] = result['fault_type']
                    ui_payload['probabilities'] = {
                        k: round(float(v), 4)
                        for k, v in result['class_probabilities'].items()
                    }

                # Push to the UI queue — non-blocking; drop if the queue is full
                try:
                    ui_queue.put_nowait(ui_payload)
                except queue.Full:
                    pass

                del batch
                del result

        recv_thread.join(timeout=2)


if __name__ == "__main__":
    # If run directly, just run the receiver without the UI. Useful for testing.
    receiver_main()