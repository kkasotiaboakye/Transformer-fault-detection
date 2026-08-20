import socket
import struct
import threading
import queue
import time
import numpy as np

from old_simulator import (
    T, CLASS_ORDER,
    make_time_axis, generate_one_healthy_signal, generate_one_faulty_signal
)

# --- Configuration (must match the receiver) ---
PI_HOST = "172.20.10.4"          # loopback: talks to receiver on same machine
PI_PORT = 65432
DTYPE = np.float32
HEADER_FORMAT = ">QIB"

FAULT_CLASSES = [c for c in CLASS_ORDER if c != 'healthy']

fault_request_queue = queue.Queue()
stop_flag = threading.Event()


def keyboard_listener():
    print("Sender ready. Streaming healthy data by default.")
    print("Type one of " + str(FAULT_CLASSES) + " + Enter to inject a fault, or 'quit' to stop.")
    while not stop_flag.is_set():
        try:
            user_input = input().strip().lower()
        except EOFError:
            break

        if user_input == "quit":
            stop_flag.set()
            break
        elif user_input == "list":
            print("Available fault types: " + str(FAULT_CLASSES))
        elif user_input in FAULT_CLASSES:
            fault_request_queue.put(user_input)
            print("Fault '" + user_input + "' queued -- will inject on next signal.")
        elif user_input == "":
            continue
        else:
            print("Unrecognized input '" + user_input + "'. Options: "
                  + str(FAULT_CLASSES) + ", 'list', 'quit'.")


def send_framed(sock, samples, running_total, label_code):
    payload = samples.astype(DTYPE).tobytes()
    running_total += len(samples)
    header = struct.pack(HEADER_FORMAT, running_total, len(payload), label_code)
    sock.sendall(header + payload)
    return running_total


def main():
    t = make_time_axis()
    rng = np.random.default_rng()
    running_total = 0
    signal_count = 0

    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    print("Connecting to receiver at " + PI_HOST + ":" + str(PI_PORT) + " ...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((PI_HOST, PI_PORT))
        print("Connected. Streaming...")

        while not stop_flag.is_set():
            loop_start = time.time()

            if not fault_request_queue.empty():
                fault_type = fault_request_queue.get()
                _, final_signal = generate_one_faulty_signal(t, fault_type, rng)
                label = fault_type
            else:
                _, final_signal = generate_one_healthy_signal(t, rng)
                label = "healthy"

            label_code = CLASS_ORDER.index(label)
            running_total = send_framed(sock, final_signal, running_total, label_code)
            signal_count += 1

            elapsed = time.time() - loop_start
            sleep_time = T - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("Sender stopped.")


if __name__ == "__main__":
    main()