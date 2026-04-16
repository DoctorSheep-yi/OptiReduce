import numpy as np
import time

def run_ring(node, grad):
    n = node.num_nodes
    flat = grad.flatten().astype(np.float32)
    chunk_size = (len(flat) + n - 1) // n
    shards = [flat[i*chunk_size : (i+1)*chunk_size] for i in range(n)]
    # Pad last shard if needed
    if len(shards[-1]) < chunk_size:
        shards[-1] = np.append(shards[-1], np.zeros(chunk_size - len(shards[-1])))

    right = (node.node_id + 1) % n

    # Scatter-Reduce (Reliable TCP)
    for step in range(n - 1):
        s_idx = (node.node_id - step) % n
        node.send(node.peers[right][0], node.peers[right][1], 
                  {"algo":"ring", "phase":"scat", "step":step, "pay":shards[s_idx]}, force_mode="tcp")
        
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "scat" and m["step"] == step:
                        incoming = node.ring_buffer.pop(i)["pay"]
                        break
            time.sleep(0.001)
        shards[(node.node_id - step - 1) % n] += incoming

    # All-Gather (Reliable TCP)
    for step in range(n - 1):
        s_idx = (node.node_id - step + 1) % n
        node.send(node.peers[right][0], node.peers[right][1], 
                  {"algo":"ring", "phase":"gath", "step":step, "pay":shards[s_idx]}, force_mode="tcp")
        
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "gath" and m["step"] == step:
                        incoming = node.ring_buffer.pop(i)["pay"]
                        break
            time.sleep(0.001)
        shards[(node.node_id - step) % n] = incoming

    return np.concatenate(shards)[:grad.size].reshape(grad.shape)