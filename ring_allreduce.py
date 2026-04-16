import numpy as np
import time

def run_ring(node, grad):
    n = node.num_nodes
    shards = np.array_split(grad.flatten(), n)
    right = (node.node_id + 1) % n

    for step in range(n - 1):
        s_idx = (node.node_id - step) % n
        print(f" [Ring] Step {step+1}: Sending shard {s_idx}")
        node.send(node.peers[right][0], node.peers[right][1], 
                  {"algo": "ring", "phase": "scat", "step": step, "pay": shards[s_idx].tolist()}, force_mode="tcp")
        
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "scat" and m["step"] == step:
                        incoming = np.array(node.ring_buffer.pop(i)["pay"]); break
            time.sleep(0.001)
        shards[(node.node_id - step - 1) % n] += incoming

    for step in range(n - 1):
        s_idx = (node.node_id - step + 1) % n
        node.send(node.peers[right][0], node.peers[right][1], 
                  {"algo": "ring", "phase": "gath", "step": step, "pay": shards[s_idx].tolist()}, force_mode="tcp")
        
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "gath" and m["step"] == step:
                        incoming = np.array(node.ring_buffer.pop(i)["pay"]); break
            time.sleep(0.001)
        shards[(node.node_id - step) % n] = incoming

    return np.concatenate(shards).reshape(grad.shape)