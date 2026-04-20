import numpy as np
import time
from globals import *

def run_ps(node, grad):
    if node.node_id == 0:
        print("[PS] Node 0 acting as server")

        with node.lock:
            node.received = [grad]

        start = time.time()

        while True:
            with node.lock:
                if len(node.received) == node.num_nodes:
                    break

            if time.time() - start > UDP_TIMEOUT:
                print("[PS] Timeout waiting for workers")
                break

            time.sleep(0.001)

        # ✅ COMPUTATION TIME
        start_comp = time.time()
        result = np.sum(node.received, axis=0)
        node.comp_time += time.time() - start_comp

        print("[PS] Aggregation done")

        for i in range(1, node.num_nodes):
            node.send(node.peers[i][0], node.peers[i][1],
                      {"algo": "ps", "phase": "pop", "payload": result})

        return result

    else:
        node.send(node.peers[0][0], node.peers[0][1],
                  {"algo": "ps", "phase": "push", "payload": grad})
        return grad