import socket
import threading
import pickle
import time
import uuid
import numpy as np

from matrix import generate_matrix, init_matrix
import PS
from noise import Noise

COORDINATOR_IP = "172.21.102.115"
COORDINATOR_PORT = 8000
PORT = 9000


class Node:
    def __init__(self):
        self.peers = []
        self.node_id = None
        self.num_nodes = None

        self.local_matrix = None
        self.received = []
        self.lock = threading.Lock()

        self.noise = Noise()

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
        print("[TCP] Listening...")

        while True:
            conn, _ = server.accept()
            threading.Thread(target=self.handle_conn, args=(conn,), daemon=True).start()

    def handle_conn(self, conn):
        data = b''
        while True:
            packet = conn.recv(65536)
            if not packet:
                break
            data += packet

        try:
            msg = pickle.loads(data)
            self.handle_message(msg)
        except Exception as e:
            print("[ERROR]", e)

        conn.close()

    # -------------------------
    # MESSAGE HANDLER
    # -------------------------
    def handle_message(self, msg):
        msg_type = msg.get("type")

        if msg_type == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(self.peers)

            print(f"[Node {self.node_id}] Peers: {self.peers}")

        elif msg_type == "RUN":
            threading.Thread(target=self.run_ps, args=(msg,), daemon=True).start()

        elif msg_type == "PUSH":
            self.handle_push(msg)

        elif msg_type == "RESULT":
            self.handle_result(msg)

        elif msg_type == "METRIC":
            # only node 0 collects
            if self.node_id == 0:
                self.collect_metric(msg)

    # -------------------------
    # SEND
    # -------------------------
    def send(self, ip, port, msg):
        payload = pickle.dumps(msg)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        sock.sendall(payload)
        sock.close()

    def broadcast(self, msg):
        for i, (ip, port) in enumerate(self.peers):
            if i == self.node_id:
                continue
            self.send(ip, port, msg)

    # -------------------------
    # PS WORKFLOW
    # -------------------------
    def run_ps(self, config):
        if config["algo"] != "PS":
            return

        size = config["matrix_size"]
        op = config["operation"]
        round_id = config["round_id"]

        print(f"\n[Node {self.node_id}] RUN PS round {round_id}")

        # -------------------------
        # 1. GENERATE (NOT TIMED)
        # -------------------------
        t0 = time.perf_counter()
        self.local_matrix = generate_matrix(size).astype(float)
        t1 = time.perf_counter()

        gen_time = (t1 - t0) * 1000

        # -------------------------
        # 2. NOISE (optional, not timed)
        # -------------------------
        self.noise.apply_straggler()

        # -------------------------
        # 3. START TIMING (ONLY ALGO)
        # -------------------------
        start_time = time.perf_counter()

        # -------------------------
        # PARAMETER SERVER LOGIC
        # -------------------------
        if self.node_id == 0:
            with self.lock:
                self.received = [self.local_matrix]

            # wait for all nodes
            while True:
                with self.lock:
                    if len(self.received) == self.num_nodes:
                        break
                time.sleep(0.001)

            # compute
            if op == "sum":
                result = PS.sum(self.received)
            else:
                result = PS.multiply(self.received)

            algo_time = (time.perf_counter() - start_time) * 1000

            print(f"[Node 0] GEN={gen_time:.2f} ms | ALGO={algo_time:.2f} ms")

            # broadcast result
            msg = {
                "type": "RESULT",
                "data": result.tolist(),
                "round_id": round_id,
                "algo_time": algo_time,
                "gen_time": gen_time
            }
            self.broadcast(msg)

            # handle locally
            self.handle_result(msg)

        else:
            # send to node 0
            ip, port = self.peers[0]

            msg = {
                "type": "PUSH",
                "data": self.local_matrix.tolist(),
                "from": self.node_id,
                "round_id": round_id
            }

            self.send(ip, port, msg)

    # -------------------------
    # HANDLE PUSH
    # -------------------------
    def handle_push(self, msg):
        data = np.array(msg["data"])

        with self.lock:
            self.received.append(data)

    # -------------------------
    # HANDLE RESULT
    # -------------------------
    def handle_result(self, msg):
        result = np.array(msg["data"])
        algo_time = msg["algo_time"]
        gen_time = msg["gen_time"]
        round_id = msg["round_id"]

        print(f"[Node {self.node_id}] RESULT received (round {round_id})")

        metric = {
            "type": "METRIC",
            "node_id": self.node_id,
            "algo_time": algo_time,
            "gen_time": gen_time,
            "round_id": round_id
        }

        if self.node_id != 0:
            ip, port = self.peers[0]
            self.send(ip, port, metric)
        else:
            self.collect_metric(metric)

    # -------------------------
    # METRIC COLLECTION
    # -------------------------
    def collect_metric(self, msg):
        print(
            f"[METRIC] Node {msg['node_id']} | "
            f"ALGO={msg['algo_time']:.2f} ms | "
            f"GEN={msg['gen_time']:.2f} ms"
        )


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    node = Node()
    
    init_matrix(seed=123 + int(time.time()) % 1000)

    node.start_server()
    time.sleep(1)
    node.register()

    print("Node started. Waiting for RUN command...")

    while True:
        time.sleep(10)