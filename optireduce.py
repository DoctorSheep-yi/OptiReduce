import numpy as np


def run_optireduce(node, grad):
    # placeholder structure

    # shard
    parts = np.array_split(grad, node.num_nodes)

    # send to all peers (TAR style)
    for i, (ip, port) in enumerate(node.peers):
        if i == node.node_id:
            continue

        msg = {
            "type": "DATA",
            "algo": "optireduce",
            "phase": "shard",
            "payload": parts[i].tolist()
        }

        node.send(ip, port, msg)

    # wait + aggregate (simplified)
    # you plug your existing logic :contentReference[oaicite:1]{index=1} here

    return grad