import numpy as np
import time
import pickle
import uuid

from globals import *

# -------------------------
# AUTO SEND (RELIANT ON NODE)
# -------------------------
def send_auto(node, ip, port, base_msg, obj):
    # We pass the object directly. node.send will handle 
    # the pickling and chunking if it's too large.
    msg = base_msg.copy()
    msg.update({
        "chunked": False,
        "payload": obj
    })
    node.send(ip, port, msg)

# -------------------------
# PARAMETER SERVER
# -------------------------
def run_ps(node, grad):
    if node.node_id == 0:
        print("[PS] Node 0 acting as server")

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
            "phase": "result"
        }

        for i, (ip, port) in enumerate(node.peers):
            if i == node.node_id:
                continue
            send_auto(node, ip, port, msg, result)

        return result

    else:
        print(f"[PS] Node {node.node_id} sending gradient")

        ip, port = node.peers[0]

        msg = {
            "type": "DATA",
            "algo": "ps",
            "phase": "push"
        }

        send_auto(node, ip, port, msg, grad)
        return None