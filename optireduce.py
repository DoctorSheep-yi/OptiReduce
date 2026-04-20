import numpy as np
import time
from globals import *

def run_optireduce(node, grad):
    n = node.num_nodes
    shards = np.array_split(grad.flatten(), n)
    my_id = node.node_id

    local_piece = shards[my_id].copy()

    # send shards
    for i in range(n):
        if i == my_id:
            continue
        msg = {
            "algo": "optireduce",
            "phase": "shard",
            "shard_id": i,
            "payload": shards[i].tolist()
        }
        node.send(node.peers[i][0], node.peers[i][1], msg)

    # receive with timeout
    received = 0
    start = time.time()

    while received < (n - 1):
        if time.time() - start > UDP_TIMEOUT_SHORT:
            print("[OptiReduce] UDP timeout during shard phase")
            break

        found = None
        with node.lock:
            if node.opti_buffer:
                msg = node.opti_buffer.pop(0)
                if msg["phase"] == "shard" and msg["shard_id"] == my_id:
                    found = np.array(msg["payload"])

        if found is not None:
            local_piece += found
            received += 1
        else:
            time.sleep(0.001)

    # approximate missing shards (paper-style idea)
    if received < (n - 1):
        missing = (n - 1) - received
        print(f"[OptiReduce] Missing {missing} shards → approximating")
        local_piece *= n / (received + 1)

    # broadcast aggregated shard
    for i in range(n):
        if i == my_id:
            continue
        msg = {
            "algo": "optireduce",
            "phase": "agg",
            "shard_id": my_id,
            "payload": local_piece.tolist()
        }
        node.send(node.peers[i][0], node.peers[i][1], msg)

    final_shards = {my_id: local_piece}
    start = time.time()

    while len(final_shards) < n:
        if time.time() - start > UDP_TIMEOUT_SHORT:
            print("[OptiReduce] UDP timeout during gather phase")
            break

        with node.lock:
            if node.opti_buffer:
                msg = node.opti_buffer.pop(0)
                if msg["phase"] == "agg":
                    final_shards[msg["shard_id"]] = np.array(msg["payload"])

        time.sleep(0.001)

    # fill missing shards with zeros (or last known)
    for i in range(n):
        if i not in final_shards:
            final_shards[i] = np.zeros_like(shards[i])

    res = [final_shards[i] for i in range(n)]
    return np.concatenate(res).reshape(grad.shape)