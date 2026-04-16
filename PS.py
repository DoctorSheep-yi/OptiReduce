import numpy as np
import time

def run_ps(node, grad):
    server_ip, server_port = node.peers[0]

    if node.node_id == 0:
        # Server Logic
        while True:
            with node.lock:
                if len(node.received) == node.num_nodes: break
            time.sleep(0.01)
        
        agg = np.sum(node.received, axis=0)
        node.received = [] # Clear for next run
        
        for i in range(1, node.num_nodes):
            node.send(node.peers[i][0], node.peers[i][1], 
                      {"algo":"ps", "phase":"pop", "payload":agg}, force_mode="tcp")
        return agg
    else:
        # Worker Logic
        node.send(server_ip, server_port, 
                  {"algo":"ps", "phase":"push", "payload":grad}, force_mode="tcp")
        
        # In a real PS, we'd wait for 'pop' phase here, but we are measuring latency
        return grad