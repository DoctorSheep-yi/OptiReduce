import numpy as np
import time

def run_optireduce(node, grad):
    n = node.num_nodes
    shards = np.array_split(grad.flatten(), n)
    my_shard_id = node.node_id
    local_piece = shards[my_shard_id].copy()
    T_BOUND = 0.5 

    # 1. Scatter (UDP)
    for i in range(n):
        if i == node.node_id: continue
        msg = {"algo": "optireduce", "phase": "shard", "shard_id": i, "payload": shards[i]}
        node.send(node.peers[i][0], node.peers[i][1], msg, force_mode="udp")

    # 2. Aggregation with Timeout
    start = time.time()
    received = 0
    while received < (n - 1) and (time.time() - start) < T_BOUND:
        found = None
        with node.lock:
            for i, m in enumerate(node.opti_buffer):
                if m.get("phase") == "shard" and m["shard_id"] == my_shard_id:
                    found = node.opti_buffer.pop(i)["payload"]; break
        if found is not None:
            local_piece += found; received += 1
        else: time.sleep(0.001)

    if received < (n - 1):
        print(f"[OptiReduce] Aggregation TIMEOUT. Only got {received}/{n-1} shards.")

    # 3. Broadcast
    for i in range(n):
        if i == node.node_id: continue
        msg = {"algo": "optireduce", "phase": "agg", "shard_id": my_shard_id, "payload": local_piece}
        node.send(node.peers[i][0], node.peers[i][1], msg, force_mode="udp")

    # 4. Gather with Zero-filling
    final_shards = {my_shard_id: local_piece}
    start = time.time()
    while len(final_shards) < n and (time.time() - start) < T_BOUND:
        with node.lock:
            for i, m in enumerate(node.opti_buffer):
                if m.get("phase") == "agg" and m["shard_id"] not in final_shards:
                    msg = node.opti_buffer.pop(i)
                    final_shards[msg["shard_id"]] = msg["payload"]; break
        time.sleep(0.001)

    res = [final_shards.get(i, np.zeros_like(shards[i])) for i in range(n)]
    return np.concatenate(res).reshape(grad.shape)