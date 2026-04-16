import numpy as np
import time

def run_ring(node, grad):
    n = node.num_nodes
    flat = grad.flatten()
    shards = np.array_split(flat, n)
    right = (node.node_id + 1) % n

    # Step 1: Scatter-Reduce (TCP)
    for step in range(n - 1):
        s_idx = (node.node_id - step) % n
        node.send(node.peers[right][0], node.peers[right][1], 
                  {"algo": "ring", "phase": "scat", "step": step, "pay": shards[s_idx]}, force_mode="tcp")
        
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "scat" and m["step"] == step:
                        incoming = node.ring_buffer.pop(i)["pay"]; break
            time.sleep(0.001)
        shards[(node.node_id - step - 1) % n] += incoming

    # Step 2: All-Gather (TCP)
    for step in range(n - 1):
        s_idx = (node.node_id - step + 1) % n
        node.send(node.peers[right][0], node.peers[right][1], 
                  {"algo": "ring", "phase": "gath", "step": step, "pay": shards[s_idx]}, force_mode="tcp")
        
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "gath" and m["step"] == step:
                        incoming = node.ring_buffer.pop(i)["pay"]; break
            time.sleep(0.001)
        shards[(node.node_id - step) % n] = incoming

    return np.concatenate(shards).reshape(grad.shape)