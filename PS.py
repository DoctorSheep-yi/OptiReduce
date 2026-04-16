import numpy as np
import time


def run_ps(node, grad):
    if node.node_id == 0:
        collected = [grad]

        while len(collected) < node.num_nodes:
            time.sleep(0.001)

        result = np.zeros_like(grad)
        for g in collected:
            result += g

        msg = {
            "type": "DATA",
            "algo": "ps",
            "phase": "result",
            "payload": result.tolist()
        }

        node.broadcast(msg)
        return result

    else:
        ip, port = node.peers[0]

        msg = {
            "type": "DATA",
            "algo": "ps",
            "phase": "push",
            "payload": grad.tolist()
        }

        node.send(ip, port, msg)
        return None