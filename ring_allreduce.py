import numpy as np
import time

def split_into_shards(flat, n):
    leftover = len(flat) % n
    if leftover != 0:
        flat = np.concatenate([flat, np.zeros(n - leftover)])
    size = len(flat) // n
    return [flat[i*size:(i+1)*size] for i in range(n)]

def send(node, target_id, msg):
    ip, port = node.peers[target_id]
    msg["chunked"] = False
    node.send(ip, port, msg)

def run_ring(node, grad):
    # Match the np.float32 type from your node.py to save bandwidth
    flat = grad.flatten().astype(np.float32)
    shards = split_into_shards(flat, node.num_nodes)

    n = node.num_nodes
    right = (node.node_id + 1) % n
    left  = (node.node_id - 1) % n

    print(f"[Ring] Node {node.node_id} starting Scatter-Reduce")
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
            "payload": shards[send_idx] # Send Numpy Array Directly
        }
        #print(f"[Ring] Node {node.node_id} sending shard {send_idx} to Node {right}")
        send(node, right, msg)

        # Safely wait for specific incoming shard
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "scatter" and m["step"] == step:
                        incoming = node.ring_buffer.pop(i)["payload"]
                        break
            if incoming is None:
                time.sleep(0.001)

        shards[recv_idx] += incoming

    print(f"[Ring] Node {node.node_id} starting All-Gather")
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
            "payload": shards[send_idx] # Send Numpy Array Directly
        }

        send(node, right, msg)

        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "gather" and m["step"] == step:
                        incoming = node.ring_buffer.pop(i)["payload"]
                        break
            if incoming is None:
                time.sleep(0.001)

        shards[recv_idx] = incoming

    print(f"[Ring] Node {node.node_id} Aggregation done")
    # reconstruct
    result = np.concatenate(shards)[:grad.size]
    return result.reshape(grad.shape)