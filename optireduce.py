import numpy as np
import time


def run_optireduce(node, grad):
    n = node.num_nodes

    flat = grad.flatten().astype(np.float64)
    shards = np.array_split(flat, n)

    my_shard_id = node.node_id
    local_piece = shards[my_shard_id].copy()

    # -------------------------
    # SEND shards to all peers
    # -------------------------
    for i in range(n):
        if i == node.node_id:
            continue

        msg = {
            "type": "DATA",
            "algo": "optireduce",
            "phase": "shard",
            "src": node.node_id,
            "shard_id": i,
            "payload": shards[i].tolist()
        }

        ip, port = node.peers[i]
        node.send(ip, port, msg)

    # -------------------------
    # RECEIVE & AGGREGATE
    # -------------------------
    received = 0

    while received < (n - 1):
        if hasattr(node, "opti_buffer") and node.opti_buffer:
            msg = node.opti_buffer.pop(0)

            if msg["shard_id"] == my_shard_id:
                local_piece += np.array(msg["payload"])
                received += 1
        else:
            time.sleep(0.001)

    # -------------------------
    # BROADCAST aggregated shard
    # -------------------------
    for i in range(n):
        if i == node.node_id:
            continue

        msg = {
            "type": "DATA",
            "algo": "optireduce",
            "phase": "agg",
            "src": node.node_id,
            "shard_id": my_shard_id,
            "payload": local_piece.tolist()
        }

        ip, port = node.peers[i]
        node.send(ip, port, msg)

    # -------------------------
    # GATHER all shards
    # -------------------------
    final_shards = {my_shard_id: local_piece}

    while len(final_shards) < n:
        if node.opti_buffer:
            msg = node.opti_buffer.pop(0)
            if msg["phase"] == "agg":
                sid = msg["shard_id"]
                final_shards[sid] = np.array(msg["payload"])
        else:
            time.sleep(0.001)

    result = np.concatenate([final_shards[i] for i in range(n)])
    return result.reshape(grad.shape)