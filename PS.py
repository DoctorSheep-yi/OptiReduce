import numpy as np
import time

def run_ps(node, grad):
    if node.node_id == 0:
        print("[PS] Node 0 acting as server")
        # Add local grad to the buffer manually so we don't wait for a network msg from ourselves
        with node.lock:
            node.received.append(grad.tolist())

        while True:
            with node.lock:
                if len(node.received) == node.num_nodes:
                    break
            time.sleep(0.001)

        # Reconstruct and sum
        result = np.sum([np.array(g) for g in node.received], axis=0)
        node.received = [] # Clear
        print("[PS] Aggregation done")

        for i in range(1, node.num_nodes):
            msg = {"algo": "ps", "phase": "pop", "payload": result.tolist()}
            node.send(node.peers[i][0], node.peers[i][1], msg, force_mode=None)
        return result
    else:
        print(f"[PS] Node {node.node_id} sending gradient")
        node.send(node.peers[0][0], node.peers[0][1], 
                  {"algo": "ps", "phase": "push", "payload": grad.tolist()}, force_mode=None)
        return grad