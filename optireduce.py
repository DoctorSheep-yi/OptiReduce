import numpy as np
import time

def run_optireduce(node, grad):
    n = node.num_nodes

    # Match the np.float32 type from your node.py
    flat = grad.flatten().astype(np.float32)
    shards = np.array_split(flat, n)

    my_shard_id = node.node_id
    local_piece = shards[my_shard_id].copy()

    print(f"[OptiReduce] Node {node.node_id} sending shards")
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
            "chunked": False,
            "payload": shards[i] # Send Numpy Array Directly
        }

        ip, port = node.peers[i]
        node.send(ip, port, msg)

    # -------------------------
    # RECEIVE & AGGREGATE
    # -------------------------
    received = 0

    while received < (n - 1):
        found_payload = None
        with node.lock:
            for i, m in enumerate(node.opti_buffer):
                if m["phase"] == "shard" and m["shard_id"] == my_shard_id:
                    found_payload = node.opti_buffer.pop(i)["payload"]
                    break
        
        if found_payload is not None:
            local_piece += found_payload
            received += 1
        else:
            time.sleep(0.001)

    print(f"[OptiReduce] Node {node.node_id} broadcasting aggregated shard")
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
            "chunked": False,
            "payload": local_piece # Send Numpy Array Directly
        }

        ip, port = node.peers[i]
        node.send(ip, port, msg)

    # -------------------------
    # GATHER all shards
    # -------------------------
    final_shards = {my_shard_id: local_piece}

    while len(final_shards) < n:
        found_shard_id = None
        found_payload = None
        
        with node.lock:
            for i, m in enumerate(node.opti_buffer):
                if m["phase"] == "agg" and m["shard_id"] not in final_shards:
                    msg = node.opti_buffer.pop(i)
                    found_shard_id = msg["shard_id"]
                    found_payload = msg["payload"]
                    break
                    
        if found_shard_id is not None:
            final_shards[found_shard_id] = found_payload
        else:
            time.sleep(0.001)

    print(f"[OptiReduce] Node {node.node_id} Aggregation done")
    result = np.concatenate([final_shards[i] for i in range(n)])
    return result.reshape(grad.shape)