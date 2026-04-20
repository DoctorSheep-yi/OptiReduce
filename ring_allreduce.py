import numpy as np
import time

def run_ring(node, grad):
    n = node.num_nodes
    shards = np.array_split(grad.flatten(), n)
    right = (node.node_id + 1) % n

    TIMEOUT = 10  # seconds

    # 🔥 small barrier to reduce desync
    time.sleep(1)

    print(f"[Node {node.node_id}] Ring start with {n} nodes")

    # =========================
    # SCATTER-REDUCE
    # =========================
    for step in range(n - 1):
        s_idx = (node.node_id - step) % n

        print(f"[Node {node.node_id}] SEND scat step {step} -> Node {right} (shard {s_idx})")

        node.send(
            node.peers[right][0],
            node.peers[right][1],
            {
                "algo": "ring",
                "phase": "scat",
                "step": step,
                "pay": shards[s_idx].tolist()
            },
            force_mode="tcp"
        )

        incoming = None
        start = time.time()

        while incoming is None:
            # ⛔ timeout protection
            if time.time() - start > TIMEOUT:
                print(f"[Node {node.node_id}] ❌ TIMEOUT in SCATTER step {step}")
                return grad

            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m.get("phase") == "scat" and m.get("step") == step:
                        incoming = np.array(m["pay"])
                        node.ring_buffer.pop(i)
                        print(f"[Node {node.node_id}] RECV scat step {step}")
                        break

            time.sleep(0.001)

        target_idx = (node.node_id - step - 1) % n
        shards[target_idx] += incoming

    # =========================
    # ALL-GATHER
    # =========================
    for step in range(n - 1):
        s_idx = (node.node_id - step + 1) % n

        print(f"[Node {node.node_id}] SEND gath step {step} -> Node {right} (shard {s_idx})")

        node.send(
            node.peers[right][0],
            node.peers[right][1],
            {
                "algo": "ring",
                "phase": "gath",
                "step": step,
                "pay": shards[s_idx].tolist()
            },
            force_mode="tcp"
        )

        incoming = None
        start = time.time()

        while incoming is None:
            # ⛔ timeout protection
            if time.time() - start > TIMEOUT:
                print(f"[Node {node.node_id}] ❌ TIMEOUT in GATHER step {step}")
                return grad

            with node.lock:
                for i, m in enumerate(node.ring_buffer):
                    if m.get("phase") == "gath" and m.get("step") == step:
                        incoming = np.array(m["pay"])
                        node.ring_buffer.pop(i)
                        print(f"[Node {node.node_id}] RECV gath step {step}")
                        break

            time.sleep(0.001)

        target_idx = (node.node_id - step) % n
        shards[target_idx] = incoming

    print(f"[Node {node.node_id}] Ring completed")

    return np.concatenate(shards).reshape(grad.shape)