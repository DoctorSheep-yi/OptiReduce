import socket
import threading
import pickle
import time
import numpy as np

from matrix import generate_matrix

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

            print(f"\n[Node {self.node_id}] Peers updated")

        elif msg_type == "START":
            threading.Thread(target=self.run_ps, args=(msg,), daemon=True).start()

        elif msg_type == "PUSH":
            self.handle_push(msg)

        elif msg_type == "RESULT":
            self.handle_result(msg)

    # -------------------------
    # SEND HELPERS
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
    # PS EXECUTION
    # -------------------------
    def run_ps(self, msg):
        size = msg["size"]

        print(f"[Node {self.node_id}] Start round with size={size}")

        # -------------------------
        # GENERATE (NOT TIMED)
        # -------------------------
        try:
            self.local_matrix = generate_matrix(size)
        except Exception as e:
            print(f"[Node {self.node_id}] ERROR:", e)
            return

        # -------------------------
        # START TIMING (ALGO ONLY)
        # -------------------------
        start_time = time.perf_counter()

        if self.node_id == 0:
            # server
            with self.lock:
                self.received = [self.local_matrix]

            # wait all workers
            while True:
                with self.lock:
                    if len(self.received) == self.num_nodes:
                        break
                time.sleep(0.001)

            # aggregate (SUM)
            result = np.zeros_like(self.local_matrix)
            for m in self.received:
                result += m

            latency = (time.perf_counter() - start_time) * 1000

            print(f"[Server] Done. Latency={latency:.2f} ms")

            # broadcast result
            msg = {
                "type": "RESULT",
                "data": result.tolist(),
                "latency": latency
            }
            self.broadcast(msg)

            # also print locally
            self.handle_result(msg)

        else:
            # worker → send to server
            ip, port = self.peers[0]

            msg = {
                "type": "PUSH",
                "data": self.local_matrix.tolist(),
                "from": self.node_id
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
        latency = msg["latency"]
        print(f"[Node {self.node_id}] RESULT received. Latency={latency:.2f} ms")

    # -------------------------
    # CLI (ONLY SERVER USES)
    # -------------------------
    def cli_loop(self):
        while True:
            cmd = input(">> ").strip()

            if cmd.startswith("start"):
                if self.node_id != 0:
                    print("Only node 0 can start!")
                    continue

                try:
                    size = int(cmd.split()[1])
                except:
                    print("Usage: start <size>")
                    continue

                msg = {
                    "type": "START",
                    "size": size
                }

                print(f"[Server] Broadcasting size={size}")
                self.broadcast(msg)

                # also run locally
                self.run_ps(msg)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    node = Node()
    node.start_server()

    time.sleep(1)
    node.register()

    print("Node started. Waiting for peers...")

    # wait for peer update
    while node.node_id is None:
        time.sleep(1)

    print(f"[Node {node.node_id}] Ready")

    # only server has CLI
    if node.node_id == 0:
        node.cli_loop()
    else:
        while True:
            time.sleep(10)