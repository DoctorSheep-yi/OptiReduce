import numpy as np
import time

def run_optireduce(node, grad):
    n = node.num_nodes
    shards = np.array_split(grad.flatten(), n)
    my_shard_id = node.node_id
    local_piece = shards[my_shard_id].copy()

    for i in range(n):
        if i == node.node_id: continue
        msg = {"algo": "optireduce", "phase": "shard", "shard_id": i, "payload": shards[i].tolist()}
        node.send(node.peers[i][0], node.peers[i][1], msg, force_mode="udp")

    received = 0
    while received < (n - 1):
        found = None
        with node.lock:
            if node.opti_buffer:
                msg = node.opti_buffer.pop(0)
                if msg["shard_id"] == my_shard_id:
                    found = np.array(msg["payload"])
        if found is not None:
            local_piece += found; received += 1
        else: time.sleep(0.001)

    for i in range(n):
        if i == node.node_id: continue
        msg = {"algo": "optireduce", "phase": "agg", "shard_id": my_shard_id, "payload": local_piece.tolist()}
        node.send(node.peers[i][0], node.peers[i][1], msg, force_mode="udp")

    final_shards = {my_shard_id: local_piece}
    while len(final_shards) < n:
        with node.lock:
            if node.opti_buffer:
                msg = node.opti_buffer.pop(0)
                if msg["phase"] == "agg":
                    final_shards[msg["shard_id"]] = np.array(msg["payload"])
        time.sleep(0.001)

    res = [final_shards.get(i, np.zeros_like(shards[i])) for i in range(n)]
    return np.concatenate(res).reshape(grad.shape)