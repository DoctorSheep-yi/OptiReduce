import numpy as np
import time


def run_ps(node, grad):
    if node.node_id == 0:
        print("[PS] Node 0 acting as server")

        # SAME variable as original node.py
        node.received = [grad]

        while True:
            if len(node.received) == node.num_nodes:
                break
            time.sleep(0.001)

        result = np.zeros_like(grad)
        for g in node.received:
            result += g

        print("[PS] Aggregation done")

        msg = {
            "type": "DATA",
            "algo": "ps",
            "phase": "result",
            "payload": result.tolist()
        }

        node.broadcast(msg)
        return result

    else:
        print(f"[PS] Node {node.node_id} sending gradient")

        ip, port = node.peers[0]

        msg = {
            "type": "DATA",
            "algo": "ps",
            "phase": "push",
            "payload": grad.tolist()
        }

        node.send(ip, port, msg)
        return None