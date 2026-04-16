import socket
import threading
import pickle
import time
import numpy as np

from matrix import generate_matrix

from PS import run_ps
from ring_allreduce import run_ring
from optireduce import run_optireduce

from globals import *

COORDINATOR_IP = "172.21.102.115"
COORDINATOR_PORT = 8000
PORT = 9000

class Node:
    def __init__(self, mode):
        self.mode = mode
        self.peers = []
        self.node_id = None
        self.num_nodes = None

        self.local_matrix = None

        self.done_count = 0
        self.lock = threading.Lock()

        self.algo = None
        self.start_time = None

        # shared buffers
        self.ring_buffer = []
        self.opti_buffer = []
        self.received = []
        self.chunk_buffer = {}

    # -------------------------
    # NETWORK SEND
    # -------------------------
    def _send_chunked(self, ip, port, msg):
        data = pickle.dumps(msg)

        chunk_id = str(uuid.uuid4())
        chunk_size = 800

        total = (len(data) + chunk_size - 1) // chunk_size

        for i in range(total):
            chunk = data[i * chunk_size:(i + 1) * chunk_size]

            chunk_msg = {
                "chunked": True,
                "chunk_id": chunk_id,
                "chunk_idx": i,
                "num_chunks": total,
                "payload": chunk
            }

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(pickle.dumps(chunk_msg), (ip, port))
            sock.close()

            time.sleep(0.0005)

    def send(self, ip, port, msg):
        data = pickle.dumps(msg)

        if self.mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            sock.sendall(data)
            sock.close()

        else:
            if len(data) <= MAX_CHUNK_SIZE:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(data, (ip, port))
                sock.close()
            else:
                self._send_chunked(ip, port, msg)

        def broadcast(self, msg):
            print(f"[Node {self.node_id}] Broadcasting {msg['type']} ({self.mode})")
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
    # SERVER START
    # -------------------------
    def start_server(self):
        threading.Thread(target=self.tcp_server, daemon=True).start()

        if self.mode == "udp":
            threading.Thread(target=self.udp_server, daemon=True).start()

    # -------------------------
    # TCP SERVER
    # -------------------------
    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", PORT))
        server.listen()

        print("[TCP] Listening...")

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
            print("[TCP ERROR]", e)

    # -------------------------
    # UDP SERVER (🔥 FIX)
    # -------------------------
    def udp_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", PORT))

        print("[UDP] Listening...")

        while True:
            data, _ = sock.recvfrom(65536)
            try:
                msg = pickle.loads(data)
                self.handle_message(msg)
            except Exception as e:
                print("[UDP ERROR]", e)

    # -------------------------
    # MESSAGE HANDLER
    # -------------------------
    def handle_message(self, msg):
        t = msg.get("type")

        # ---------- CHUNK HANDLING ----------
        if msg.get("chunked", False):

            cid = msg["chunk_id"]

            if cid not in self.chunk_buffer:
                self.chunk_buffer[cid] = {
                    "chunks": {},
                    "total": msg["num_chunks"]
                }

            self.chunk_buffer[cid]["chunks"][msg["chunk_idx"]] = msg["payload"]

            if len(self.chunk_buffer[cid]["chunks"]) == msg["num_chunks"]:
                chunks = self.chunk_buffer[cid]["chunks"]

                try:
                    data = b"".join(chunks[i] for i in range(msg["num_chunks"]))
                except KeyError:
                    print("[CHUNK ERROR] Missing chunk")
                    return

                full_payload = pickle.loads(data)

                msg["payload"] = full_payload
                msg["chunked"] = False  # now it's normal
                del self.chunk_buffer[cid]

            else:
                return  # wait for more chunks

        # ---------- coordinator ----------
        if t == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(self.peers)
            print(f"[Node {self.node_id}] Peers updated")
            return

        # ---------- START ----------
        if t == "START":
            print(f"[Node {self.node_id}] Received START")
            threading.Thread(target=self.run_experiment, args=(msg,), daemon=True).start()
            return

        # ---------- DONE ----------
        if t == "DONE":
            if self.node_id == 0:
                with self.lock:
                    self.done_count += 1
            return

        # ---------- ALGO ----------
        algo = msg.get("algo")

        if algo == "ring":
            self.ring_buffer.append(np.array(msg["payload"]))
            return

        if algo == "optireduce":
            self.opti_buffer.append(msg)
            return

        if algo == "ps":
            self.handle_ps(msg)
            return

    # -------------------------
    # PS HANDLER
    # -------------------------
    def handle_ps(self, msg):
        phase = msg.get("phase")

        if phase == "push":
            data = msg["payload"]   # ✅ IMPORTANT FIX (no np.array)

            with self.lock:
                self.received.append(data)

            print(f"[PS] Node {self.node_id} received PUSH "
                f"({len(self.received)}/{self.num_nodes})")

        elif phase == "result":
            print(f"[Node {self.node_id}] Received FINAL RESULT")

    # -------------------------
    # EXPERIMENT
    # -------------------------
    def run_experiment(self, msg):
        self.algo = msg["algo"]
        size = msg["size"]

        print(f"[Node {self.node_id}] START {self.algo}, size={size}")

        # ---------- NOT TIMED ----------
        print(f"[Node {self.node_id}] Generating matrix...")
        self.local_matrix = generate_matrix(size).astype(np.float32)

        print(f"[Node {self.node_id}] Computing gradient...")
        gradient = np.tanh(self.local_matrix)

        time.sleep(1)  # barrier

        if self.node_id == 0:
            self.done_count = 0
            self.start_time = time.perf_counter()

        # ---------- RUN ----------
        if self.algo == "ps":
            result = run_ps(self, gradient)

        elif self.algo == "ring":
            result = run_ring(self, gradient)

        elif self.algo == "optireduce":
            result = run_optireduce(self, gradient)

        else:
            raise ValueError("Unknown algo")

        # ---------- DONE ----------
        if self.node_id != 0:
            ip, port = self.peers[0]
            self.send(ip, port, {"type": "DONE"})
        else:
            with self.lock:
                self.done_count += 1

            while True:
                with self.lock:
                    if self.done_count == self.num_nodes:
                        break
                time.sleep(0.001)

            latency = (time.perf_counter() - self.start_time) * 1000
            print(f"\n[FINAL] {self.algo} latency = {latency:.2f} ms\n")

    # -------------------------
    # CLI (NODE 0 ONLY)
    # -------------------------
    def cli(self):
        while True:
            cmd = input(">> ").strip().split()

            if not cmd:
                continue

            if cmd[0] == "start":
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

    node.cli()