import socket
import threading
import pickle
import time
import numpy as np

from matrix import generate_matrix

from PS import run_ps
from ring_allreduce import run_ring
from optireduce import run_optireduce

COORDINATOR_IP = "172.21.102.115"
COORDINATOR_PORT = 8000
PORT = 9000


class Node:
    def __init__(self, mode):
        self.mode = mode  # tcp | udp
        self.peers = []
        self.node_id = None
        self.num_nodes = None

        self.local_matrix = None

    # -------------------------
    # NETWORK
    # -------------------------
    def send(self, ip, port, msg):
        data = pickle.dumps(msg)

        if self.mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            sock.sendall(data)
            sock.close()
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(data, (ip, port))
            sock.close()

    def broadcast(self, msg):
        for i, (ip, port) in enumerate(self.peers):
            if i == self.node_id:
                continue
            self.send(ip, port, msg)

    # -------------------------
    # REGISTER
    # -------------------------
    def register(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((COORDINATOR_IP, COORDINATOR_PORT))
        msg = {"type": "REGISTER", "port": PORT}
        sock.sendall(pickle.dumps(msg))
        sock.close()

    # -------------------------
    # SERVER
    # -------------------------
    def start_server(self):
        threading.Thread(target=self.tcp_server, daemon=True).start()

    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", PORT))
        server.listen()

        print("[Node] Listening...")

        while True:
            conn, _ = server.accept()
            threading.Thread(target=self.handle_conn, args=(conn,), daemon=True).start()

    def handle_conn(self, conn):
        data = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk

        conn.close()

        try:
            msg = pickle.loads(data)
            self.handle_message(msg)
        except Exception as e:
            print("[ERROR]", e)

    # -------------------------
    # MESSAGE HANDLER
    # -------------------------
    def handle_message(self, msg):
        t = msg.get("type")

        if t == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(self.peers)

            print(f"[Node {self.node_id}] Peers updated")
            return

        if t == "START":
            threading.Thread(target=self.run_experiment, args=(msg,), daemon=True).start()
            return

        # algorithm messages forwarded
        if self.algo_handler:
            self.algo_handler(msg)

    # -------------------------
    # EXPERIMENT
    # -------------------------
    def run_experiment(self, msg):
        algo = msg["algo"]
        size = msg["size"]

        print(f"[Node {self.node_id}] START {algo}, size={size}")

        # ---------- NOT TIMED ----------
        self.local_matrix = generate_matrix(size).astype(np.float32)
        gradient = np.tanh(self.local_matrix)
        # --------------------------------

        # barrier (simple)
        time.sleep(1)

        start = time.perf_counter()

        if algo == "ps":
            result = run_ps(self, gradient)

        elif algo == "ring":
            result = run_ring(self, gradient)

        elif algo == "optireduce":
            result = run_optireduce(self, gradient)

        else:
            raise ValueError("Unknown algo")

        latency = (time.perf_counter() - start) * 1000

        print(f"[Node {self.node_id}] DONE {algo} latency={latency:.2f} ms")

    # -------------------------
    # CLI (ONLY NODE 0)
    # -------------------------
    def cli(self):
        while True:
            cmd = input(">> ").strip().split()

            if not cmd:
                continue

            if cmd[0] == "start":
                # start <algo> <size>
                algo = cmd[1]
                size = int(cmd[2])

                msg = {
                    "type": "START",
                    "algo": algo,
                    "size": size
                }

                self.broadcast(msg)
                self.run_experiment(msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tcp", "udp"], required=True)
    args = parser.parse_args()

    node = Node(args.mode)
    node.start_server()

    print("type 'register'")
    input("> ")
    node.register()

    time.sleep(2)

    if True:
        node.cli()