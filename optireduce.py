import numpy as np
import time
from globals import *


# =========================
# Utilities
# =========================

def next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def fwht(x):
    """In-place Fast Walsh–Hadamard Transform"""
    h = 1
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


def hadamard_transform(x):
    """Normalized Hadamard"""
    x = x.copy()
    fwht(x)
    x /= np.sqrt(len(x))   # ✅ FIX
    return x


def inverse_hadamard_transform(x):
    """Inverse = same transform"""
    x = x.copy()
    fwht(x)
    x /= np.sqrt(len(x))   # ✅ FIX
    return x


def pad_to_power_of_two(x):
    n = len(x)
    n2 = next_power_of_two(n)
    if n2 == n:
        return x, n
    padded = np.zeros(n2, dtype=x.dtype)
    padded[:n] = x
    return padded, n


def topk_sparsify(x, ratio=0.1):
    """Keep top-k magnitude values"""
    k = max(1, int(len(x) * ratio))
    idx = np.argsort(np.abs(x))[-k:]
    out = np.zeros_like(x)
    out[idx] = x[idx]
    return out


# =========================
# MAIN OPTIREDUCE
# =========================

def run_optireduce(node, grad):
    UDP_TIMEOUT_SHORT = 10
    UDP_TIMEOUT_LONG = 30
    n = node.num_nodes
    my_id = node.node_id

    flat_grad = grad.flatten()

    print(f"[Node {my_id}] OptiReduce start with {n} nodes")

    # ===== Hadamard encode =====
    padded, original_len = pad_to_power_of_two(flat_grad)

    start_comp = time.time()
    encoded = hadamard_transform(padded)
    node.comp_time += time.time() - start_comp

    # ===== Split =====
    shards = np.array_split(encoded, n)
    local_piece = shards[my_id].copy()

    # =========================
    # PHASE 1: SCATTER
    # =========================
    for i in range(n):
        if i == my_id:
            continue

        print(f"[Node {my_id}] SEND shard -> Node {i} (shard_id={i})")

        node.send(node.peers[i][0], node.peers[i][1], {
            "algo": "optireduce",
            "phase": "shard",
            "shard_id": i,
            "payload": shards[i].tolist()
        })

    # ===== RECEIVE SHARDS =====
    received = 0
    start = time.time()

    while received < (n - 1) and (time.time() - start) < UDP_TIMEOUT_SHORT:
        node.noise.apply_straggler()
        found = None

        with node.lock:
            for i, msg in enumerate(node.opti_buffer):
                if msg.get("phase") == "shard" and msg.get("shard_id") == my_id:
                    payload = np.array(msg["payload"])

                    if payload.shape == local_piece.shape:
                        found = payload
                        node.opti_buffer.pop(i)

                        print(f"[Node {my_id}] RECV shard from Node ? (shard_id={my_id})")
                        break

        if found is not None:
            start_comp = time.time()
            node.noise.apply_straggler() 
            local_piece += found
            node.comp_time += time.time() - start_comp
            received += 1
        else:
            time.sleep(0.001)

    print(f"[Node {my_id}] Scatter done, received {received}/{n-1}")

    # =========================
    # PHASE 2: ALL-GATHER
    # =========================
    for i in range(n):
        if i == my_id:
            continue

        print(f"[Node {my_id}] SEND agg -> Node {i}")

        node.send(node.peers[i][0], node.peers[i][1], {
            "algo": "optireduce",
            "phase": "agg",
            "shard_id": my_id,
            "payload": local_piece.tolist()
        })

    final_shards = {my_id: local_piece.tolist()}
    start = time.time()

    while len(final_shards) < n and (time.time() - start) < UDP_TIMEOUT_LONG:
        node.noise.apply_straggler()
        with node.lock:
            for i, msg in enumerate(node.opti_buffer):
                if msg.get("phase") == "agg":
                    sender = msg["shard_id"]
                    final_shards[sender] = np.array(msg["payload"])
                    node.opti_buffer.pop(i)

                    print(f"[Node {my_id}] RECV agg shard from Node {sender}")
                    break
        time.sleep(0.001)

    print(f"[Node {my_id}] All-gather done ({len(final_shards)}/{n})")

    # =========================
    # RECONSTRUCT
    # =========================
    full_encoded = []

    for i in range(n):
        shard = final_shards.get(i, shards[i])

        if isinstance(shard, list):
            shard = np.array(shard)

        full_encoded.append(shard)

    full_encoded = np.concatenate(full_encoded)

    # ===== Average =====
    effective_nodes = received + 1
    full_encoded = full_encoded / max(1, effective_nodes)

    # ===== Decode =====
    start_comp = time.time()
    decoded = inverse_hadamard_transform(full_encoded)   # ✅ FIXED
    node.comp_time += time.time() - start_comp

    decoded = decoded[:original_len]

    approx = decoded.reshape(grad.shape)
    truth = grad.copy()

    print(f"[Node {my_id}] OptiReduce completed")

    return approx, truth