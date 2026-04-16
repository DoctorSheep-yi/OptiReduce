import numpy as np
import time

def run_optireduce(node, grad):
    n = node.num_nodes
    flat = grad.flatten().astype(np.float32)
    shards = np.array_split(flat, n)
    my_shard_id = node.node_id
    local_piece = shards[my_shard_id].copy()
    
    # The Bounded Wait Time (t_B) from the paper
    T_BOUND = 0.8 

    # 1. Best-effort Send (UDP)
    for i in range(n):
        if i == node.node_id: continue
        msg = {"algo": "optireduce", "phase": "shard", "shard_id": i, "payload": shards[i]}
        node.send(node.peers[i][0], node.peers[i][1], msg, force_mode="udp")

    # 2. Aggregation with Timeout
    start_wait = time.time()
    received = 0
    while received < (n - 1) and (time.time() - start_wait) < T_BOUND:
        found = None
        with node.lock:
            for i, m in enumerate(node.opti_buffer):
                if m.get("phase") == "shard" and m["shard_id"] == my_shard_id:
                    found = node.opti_buffer.pop(i)["payload"]
                    break
        if found is not None:
            local_piece += found
            received += 1
        else:
            time.sleep(0.001)

    # 3. Broadcast Aggregated Shard
    for i in range(n):
        if i == node.node_id: continue
        msg = {"algo": "optireduce", "phase": "agg", "shard_id": my_shard_id, "payload": local_piece}
        node.send(node.peers[i][0], node.peers[i][1], msg, force_mode="udp")

    # 4. Final Gather with Zero-filling for lost packets
    final_shards = {my_shard_id: local_piece}
    start_gather = time.time()
    while len(final_shards) < n and (time.time() - start_gather) < T_BOUND:
        with node.lock:
            for i, m in enumerate(node.opti_buffer):
                if m.get("phase") == "agg" and m["shard_id"] not in final_shards:
                    msg = node.opti_buffer.pop(i)
                    final_shards[msg["shard_id"]] = msg["payload"]
                    break
        time.sleep(0.001)

    # Fill in missing data with zeros so the model doesn't crash
    result_parts = []
    for i in range(n):
        if i in final_shards:
            result_parts.append(final_shards[i])
        else:
            result_parts.append(np.zeros_like(shards[i]))
            
    return np.concatenate(result_parts).reshape(grad.shape)