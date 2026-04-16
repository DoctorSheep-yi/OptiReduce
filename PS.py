import numpy as np
import time
import pickle
import uuid

from globals  import *


# -------------------------
# AUTO SEND (FAST + CHUNKED)
# -------------------------
def send_auto(node, ip, port, base_msg, obj):
    data = pickle.dumps(obj)

    if len(data) <= MAX_CHUNK_SIZE:
        # -------- FAST PATH --------
        msg = base_msg.copy()
        msg.update({
            "chunked": False,
            "payload": obj
        })
        print(f"[SEND] Direct ({len(data)} bytes)")
        node.send(ip, port, msg)

    else:
        # -------- SLOW PATH --------
        print(f"[SEND] Chunking ({len(data)} bytes)")
        send_large(node, ip, port, base_msg, obj)


# -------------------------
# CHUNKED SEND
# -------------------------
def send_large(node, ip, port, base_msg, obj):
    data = pickle.dumps(obj)

    chunk_id = str(uuid.uuid4())
    total = (len(data) + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE

    for i in range(total):
        chunk = data[i * MAX_CHUNK_SIZE:(i + 1) * MAX_CHUNK_SIZE]

        msg = base_msg.copy()
        msg.update({
            "chunked": True,
            "chunk_id": chunk_id,
            "chunk_idx": i,
            "num_chunks": total,
            "payload": chunk
        })

        print(f"[PS] Sending chunk {i+1}/{total}")
        node.send(ip, port, msg)
        time.sleep(0.0005)


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