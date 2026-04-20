# optireduce.py
import numpy as np
import time
from globals import *

def hadamard_transform(x):
    """Fast Walsh-Hadamard Transform (FWHT), in-place"""
    h = 1
    x = x.copy()
    n = len(x)

    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2

    return x


def inverse_hadamard_transform(x):
    """Inverse FWHT"""
    n = len(x)
    return hadamard_transform(x) / n

def pad_to_power_of_two(x):
    n = len(x)
    next_pow2 = 1 << (n - 1).bit_length()
    if next_pow2 == n:
        return x, n
    padded = np.zeros(next_pow2, dtype=x.dtype)
    padded[:n] = x
    return padded, n

def run_optireduce(node, grad):
    n = node.num_nodes
    my_id = node.node_id

    flat_grad = grad.flatten()

    # ===== Hadamard encode =====
    padded, original_len = pad_to_power_of_two(flat_grad)
    encoded = hadamard_transform(padded)

    shards = np.array_split(encoded, n)
    local_piece = shards[my_id].copy()

    # ===== Phase 1: Scatter =====
    for i in range(n):
        if i == my_id:
            continue
        node.send(node.peers[i][0], node.peers[i][1], {
            "algo": "optireduce",
            "phase": "shard",
            "shard_id": i,
            "payload": shards[i]
        })

    # ===== Receive shards =====
    received = 0
    start = time.time()

    while received < (n - 1) and (time.time() - start) < UDP_TIMEOUT_SHORT:
        found = None
        with node.lock:
            for i, msg in enumerate(node.opti_buffer):
                if msg.get("phase") == "shard" and msg.get("shard_id") == my_id:
                    if msg["payload"].shape == local_piece.shape:
                        found = msg["payload"]
                        node.opti_buffer.pop(i)
                        break

        if found is not None:
            local_piece += found
            received += 1
        else:
            time.sleep(0.001)

    # 🚫 NO SCALING (important fix)

    # ===== Phase 2: All-gather =====
    for i in range(n):
        if i == my_id:
            continue
        node.send(node.peers[i][0], node.peers[i][1], {
            "algo": "optireduce",
            "phase": "agg",
            "shard_id": my_id,
            "payload": local_piece
        })

    final_shards = {my_id: local_piece}
    start = time.time()

    while len(final_shards) < n and (time.time() - start) < UDP_TIMEOUT_LONG:
        with node.lock:
            for i, msg in enumerate(node.opti_buffer):
                if msg.get("phase") == "agg":
                    final_shards[msg["shard_id"]] = msg["payload"]
                    node.opti_buffer.pop(i)
                    break
        time.sleep(0.001)

    # ===== Reconstruct encoded vector =====
    full_encoded = []
    shard_template = np.array_split(np.zeros_like(encoded), n)

    for i in range(n):
        full_encoded.append(final_shards.get(i, shard_template[i]))

    full_encoded = np.concatenate(full_encoded)

    # ===== Normalize (average of received pieces) =====
    effective_nodes = received + 1
    full_encoded = full_encoded / max(1, effective_nodes)

    # ===== Hadamard decode =====
    decoded = inverse_hadamard_transform(full_encoded)

    # Remove padding
    decoded = decoded[:original_len]

    approx = decoded.reshape(grad.shape)
    truth = grad.copy()

    return approx, truth
