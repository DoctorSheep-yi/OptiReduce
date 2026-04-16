import numpy as np
import time

def run_ps(node, grad):
    if node.node_id == 0:
        while True:
            with node.lock:
                if len(node.received) == node.num_nodes: break
            time.sleep(0.01)
        res = np.sum(node.received, axis=0)
        node.received = []
        for i in range(1, node.num_nodes):
            node.send(node.peers[i][0], node.peers[i][1], {"algo":"ps", "phase":"pop", "payload":res}, force_mode="tcp")
        return res
    else:
        node.send(node.peers[0][0], node.peers[0][1], {"algo":"ps", "phase":"push", "payload":grad}, force_mode="tcp")
        return grad