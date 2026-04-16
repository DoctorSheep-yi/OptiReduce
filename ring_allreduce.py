import numpy as np
import time

def run_ring(node, grad):
    n = node.num_nodes
    flat = grad.flatten().astype(np.float32)
    shards = np.array_split(flat, n)
    
    right = (node.node_id + 1) % n
    print(f"[Ring] Starting AllReduce. Neighbor: Node {right}")

    # --- Phase 1: Scatter-Reduce ---
    for step in range(n - 1):
        # Calculate which shard to send and which to receive
        send_idx = (node.node_id - step) % n
        recv_idx = (node.node_id - step - 1) % n
        
        print(f" [Scatter] Step {step+1}/{n-1}: Sending shard {send_idx} to Node {right}")
        
        msg = {"algo": "ring", "phase": "scat", "step": step, "pay": shards[send_idx]}
        node.send(node.peers[right][0], node.peers[right][1], msg, force_mode="tcp")
        
        # Wait for incoming data from left neighbor
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "scat" and m["step"] == step:
                        incoming = node.ring_buffer.pop(i)["pay"]
                        break
            time.sleep(0.001)
        
        shards[recv_idx] += incoming
    
    print(f"[Ring] Scatter-Reduce complete. Starting All-Gather.")

    # --- Phase 2: All-Gather ---
    for step in range(n - 1):
        send_idx = (node.node_id - step + 1) % n
        
        print(f" [Gather] Step {step+1}/{n-1}: Sending aggregated shard {send_idx} to Node {right}")
        
        msg = {"algo": "ring", "phase": "gath", "step": step, "pay": shards[send_idx]}
        node.send(node.peers[right][0], node.peers[right][1], msg, force_mode="tcp")
        
        incoming = None
        while incoming is None:
            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m["phase"] == "gath" and m["step"] == step:
                        incoming = node.ring_buffer.pop(i)["pay"]
                        break
            time.sleep(0.001)
        
        # Update local shard with the completed piece
        recv_idx = (node.node_id - step) % n
        shards[recv_idx] = incoming

    print(f"[Ring] All-Gather complete.")
    return np.concatenate(shards).reshape(grad.shape)