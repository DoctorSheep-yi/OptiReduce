import numpy as np
import time


def split(arr, n):
    return np.array_split(arr.flatten(), n)


def run_ring(node, grad):
    shards = split(grad, node.num_nodes)

    right = (node.node_id + 1) % node.num_nodes
    left = (node.node_id - 1) % node.num_nodes

    # scatter-reduce
    for step in range(node.num_nodes - 1):
        send_idx = (node.node_id - step) % node.num_nodes
        recv_idx = (node.node_id - step - 1) % node.num_nodes

        ip, port = node.peers[right]

        msg = {
            "type": "DATA",
            "algo": "ring",
            "phase": "scatter",
            "step": step,
            "payload": shards[send_idx].tolist()
        }

        node.send(ip, port, msg)

        # TODO: store incoming properly (simplified here)
        time.sleep(0.01)

    # all-gather similar...

    return grad