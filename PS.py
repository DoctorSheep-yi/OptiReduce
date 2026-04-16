import numpy as np
import time

def run_ps(node, grad):
    # Determine server address (Node 0)
    server_ip, server_port = node.peers[0]

    if node.node_id == 0:
        print(f"[PS Server] Waiting for gradients from {node.num_nodes} nodes...")
        
        # Wait until the received list is full
        while True:
            with node.lock:
                if len(node.received) == node.num_nodes:
                    break
            time.sleep(0.01)
        
        print(f"[PS Server] All gradients received. Aggregating...")
        result = np.sum(node.received, axis=0)
        node.received = [] # Clear for next run
        
        print(f"[PS Server] Broadcasting result to workers.")
        for i in range(1, node.num_nodes):
            msg = {"algo": "ps", "phase": "pop", "payload": result}
            node.send(node.peers[i][0], node.peers[i][1], msg, force_mode="tcp")
            
        return result
    else:
        print(f"[Node {node.node_id}] Pushing gradient to Server (Node 0).")
        push_msg = {"algo": "ps", "phase": "push", "payload": grad}
        node.send(server_ip, server_port, push_msg, force_mode="tcp")
        
        # Note: In this simplified measurement, workers finish once they've sent
        return grad