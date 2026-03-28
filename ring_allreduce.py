import socket
import pickle
import numpy as np
import time
import argparse
import sys

# import node.py directly so we reuse its peer info
from node import Node

RING_PORT = 9001


def send_shard(ip, shard, phase, step):
    msg = {
        "type"  : "RING_SHARD",
        "phase" : phase,
        "step"  : step,
        "shard" : shard.tolist(),
        "dtype" : str(shard.dtype)
    }
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, RING_PORT))
    s.sendall(pickle.dumps(msg))
    s.close()


def receive_shard(server):
    conn, addr = server.accept()
    data = b""
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
    conn.close()
    msg = pickle.loads(data)
    return np.array(msg["shard"], dtype=msg["dtype"])


def split_into_shards(flat_array, n):
    leftover = len(flat_array) % n
    if leftover != 0:
        flat_array = np.concatenate([
            flat_array,
            np.zeros(n - leftover, dtype=flat_array.dtype)
        ])
    size = len(flat_array) // n
    return [flat_array[i*size:(i+1)*size] for i in range(n)]


class RingAllReduce:

    def __init__(self, my_id, peers):
        self.my_id  = my_id
        self.peers  = peers
        self.n      = len(peers)
        self.right  = (my_id + 1) % self.n
        self.left   = (my_id - 1) % self.n

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", RING_PORT))
        self.server.listen(10)

        print(f"node {my_id} ready — left={self.left} right={self.right}")

    def run(self, matrix, method="average"):
        shape  = matrix.shape
        size   = matrix.size
        flat   = matrix.flatten().astype(np.float64)
        shards = split_into_shards(flat, self.n)

        start  = time.perf_counter()
        shards = self.scatter_reduce(shards)
        shards = self.all_gather(shards)
        ms     = (time.perf_counter() - start) * 1000

        result = np.concatenate(shards)[:size]
        if method == "average":
            result = result / self.n

        result = result.reshape(shape)
        print(f"done in {ms:.2f} ms")
        print(f"result:\n{result}")
        return result, ms

    def scatter_reduce(self, shards):
        print("--- scatter reduce ---")
        for step in range(self.n - 1):
            send_idx = (self.my_id - step)     % self.n
            recv_idx = (self.my_id - step - 1) % self.n

            right_ip, _ = self.peers[self.right]
            send_shard(right_ip, shards[send_idx], "scatter_reduce", step)
            print(f"  step {step} — sent shard[{send_idx}] to node {self.right}")

            incoming = receive_shard(self.server)
            print(f"  step {step} — got  shard[{recv_idx}] from node {self.left}")

            shards[recv_idx] = shards[recv_idx] + incoming

        return shards

    def all_gather(self, shards):
        print("--- all gather ---")
        for step in range(self.n - 1):
            send_idx = (self.my_id - step + 1) % self.n
            recv_idx = (self.my_id - step)     % self.n

            right_ip, _ = self.peers[self.right]
            send_shard(right_ip, shards[send_idx], "all_gather", step)
            print(f"  step {step} — sent shard[{send_idx}] to node {self.right}")

            incoming = receive_shard(self.server)
            print(f"  step {step} — got  shard[{recv_idx}] from node {self.left}")

            shards[recv_idx] = incoming

        return shards


if __name__ == "__main__":
    from matrix import generate_matrix

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["tcp", "udp"], required=True)
    parser.add_argument("--size",   type=int, default=4)
    parser.add_argument("--method", choices=["average", "sum"], default="average")
    args = parser.parse_args()

    # start node.py's node — this handles registration and PEER_UPDATE
    node = Node(args.mode)
    node.start_server()

    print("type 'register' then press enter")
    cmd = input("> ").strip()
    if cmd == "register":
        node.register()

    # wait until node.py gets PEER_UPDATE from coordinator
    print("waiting for peer update...")
    while node.peers is None or node.node_id is None:
        time.sleep(0.5)

    # now use the peer info node.py already has
    my_id = node.node_id
    peers = node.peers

    matrix = generate_matrix(args.size)
    print(f"my matrix:\n{matrix}")

    ring = RingAllReduce(my_id, peers)
    result, ms = ring.run(matrix, method=args.method)
