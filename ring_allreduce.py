import numpy as np
import socket
import pickle
import time

RING_PORT = 9100


def split_into_shards(flat, n):
    leftover = len(flat) % n
    if leftover != 0:
        flat = np.concatenate([flat, np.zeros(n - leftover)])
    size = len(flat) // n
    return [flat[i*size:(i+1)*size] for i in range(n)]


def send(node, target_id, msg):
    ip, port = node.peers[target_id]
    node.send(ip, port, msg)


def run_ring(node, grad):
    flat = grad.flatten().astype(np.float64)
    shards = split_into_shards(flat, node.num_nodes)

    n = node.num_nodes
    right = (node.node_id + 1) % n
    left  = (node.node_id - 1) % n

    # -------------------------
    # SCATTER-REDUCE
    # -------------------------
    for step in range(n - 1):
        send_idx = (node.node_id - step) % n
        recv_idx = (node.node_id - step - 1) % n

        msg = {
            "type": "DATA",
            "algo": "ring",
            "phase": "scatter",
            "step": step,
            "src": node.node_id,
            "idx": send_idx,
            "payload": shards[send_idx].tolist()
        }

        send(node, right, msg)

        # wait for incoming shard
        while True:
            if hasattr(node, "ring_buffer") and node.ring_buffer:
                incoming = node.ring_buffer.pop(0)
                break
            time.sleep(0.001)

        shards[recv_idx] += incoming

    # -------------------------
    # ALL-GATHER
    # -------------------------
    for step in range(n - 1):
        send_idx = (node.node_id - step + 1) % n
        recv_idx = (node.node_id - step) % n

        msg = {
            "type": "DATA",
            "algo": "ring",
            "phase": "gather",
            "step": step,
            "src": node.node_id,
            "idx": send_idx,
            "payload": shards[send_idx].tolist()
        }

        send(node, right, msg)

        while True:
            if node.ring_buffer:
                incoming = node.ring_buffer.pop(0)
                break
            time.sleep(0.001)

        shards[recv_idx] = incoming

    # reconstruct
    result = np.concatenate(shards)[:grad.size]
    return result.reshape(grad.shape)