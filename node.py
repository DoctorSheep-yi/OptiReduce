import socket
import threading
import pickle
import time
import numpy as np
import uuid
import sys

from matrix import generate_matrix
from PS import run_ps
from ring_allreduce import run_ring
from optireduce import run_optireduce
from noise import Noise
from globals import *


# =========================
# TCP helper
# =========================
def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


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

        self.noise = Noise()

        # buffers
        self.ring_buffer = []
        self.opti_buffer = []
        self.received = []
        self.chunk_buffer = {}
        self.final_result = None

    # =========================
    # SEND
    # =========================
    def send(self, ip, port, msg, force_mode=None):
        mode = force_mode if force_mode else self.mode

        if mode == "udp" and self.noise.should_drop_udp():
            return

        data = pickle.dumps(msg)

        if mode == "tcp":
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((ip, port))

                # length-prefixed framing
                sock.sendall(len(data).to_bytes(4, 'big') + data)
                sock.close()

            except Exception as e:
                print(f"[TCP ERROR] {e}")

        else:
            self._send_chunked(ip, port, data)

    def _send_chunked(self, ip, port, data):
        chunk_id = str(uuid.uuid4())
        chunk_size = MAX_CHUNK_SIZE
        total = (len(data) + chunk_size - 1) // chunk_size

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        for i in range(total):
            chunk = data[i * chunk_size:(i + 1) * chunk_size]
            msg = {
                "chunked": True,
                "chunk_id": chunk_id,
                "chunk_idx": i,
                "num_chunks": total,
                "payload": chunk
            }
            sock.sendto(pickle.dumps(msg), (ip, port))

        sock.close()

    # =========================
    # RECEIVE HANDLER
    # =========================
    def handle_message(self, msg):

        # ---- UDP reassembly ----
        if msg.get("chunked"):
            cid = msg["chunk_id"]
            if cid not in self.chunk_buffer:
                self.chunk_buffer[cid] = {}

            self.chunk_buffer[cid][msg["chunk_idx"]] = msg["payload"]

            if len(self.chunk_buffer[cid]) == msg["num_chunks"]:
                full = b''.join(
                    self.chunk_buffer[cid][i]
                    for i in range(msg["num_chunks"])
                )
                del self.chunk_buffer[cid]
                self.handle_message(pickle.loads(full))
            return

        t = msg.get("type")

        # ---- Coordinator ----
        if t == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(msg["peers"])
            print(f"[Node {self.node_id}] Ready ({self.num_nodes} nodes)")

        elif t == "START":
            threading.Thread(target=self.run_experiment, args=(msg,), daemon=True).start()

        elif t == "DONE":
            with self.lock:
                self.done_count += 1

        # ---- PS ----
        elif msg.get("algo") == "ps":
            if msg.get("phase") == "push":
                with self.lock:
                    self.received.append(msg["payload"])

            elif msg.get("phase") == "pop":
                print("[PS] Received results")
                self.final_result = np.array(msg["payload"])

        # ---- Ring ----
        elif msg.get("algo") == "ring":
            with self.lock:
                self.ring_buffer.append(msg)

        # ---- OptiReduce ----
        elif msg.get("algo") == "optireduce":
            with self.lock:
                self.opti_buffer.append(msg)

    # =========================
    # SERVERS
    # =========================
    def start_server(self, port):
        threading.Thread(target=self._tcp_server, args=(port,), daemon=True).start()
        threading.Thread(target=self._udp_server, args=(port,), daemon=True).start()

    def _tcp_server(self, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", port))
    server.listen()

    print(f"[Node] TCP server listening on {port}")

    while True:
        conn, _ = server.accept()

        try:
            length_bytes = recv_exact(conn, 4)
            if not length_bytes:
                conn.close()
                continue

            length = int.from_bytes(length_bytes, 'big')

            # sanity check (optional but good)
            if length <= 0 or length > 10_000_000:
                print(f"[TCP ERROR] Invalid length: {length}")
                conn.close()
                continue

            data = recv_exact(conn, length)
            if not data:
                conn.close()
                continue

            msg = pickle.loads(data)
            self.handle_message(msg)

        except Exception as e:
            print(f"[TCP RECV ERROR] {e}")

        conn.close()

    def _udp_server(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))

        print(f"[Node] UDP server listening on {port}")

        while True:
            data, _ = sock.recvfrom(65535)
            try:
                msg = pickle.loads(data)
                self.handle_message(msg)
            except:
                pass

    # =========================
    # COORDINATOR REGISTER
    # =========================
    def register(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((COORDINATOR_IP, COORDINATOR_PORT))

            msg = {
                "type": "REGISTER",
                "port": PORT
            }

            sock.sendall(pickle.dumps(msg))
            sock.close()

        except Exception as e:
            print(f"[Register Error] {e}")

    # =========================
    # RUN EXPERIMENT
    # =========================
    def run_experiment(self, msg):
        self.algo = msg["algo"]
        size = msg["size"]

        print(f"[Node {self.node_id}] START {self.algo}")

        self.local_matrix = generate_matrix(size)
        grad = np.tanh(self.local_matrix)

        if self.node_id == 0:
            self.done_count = 0
            self.start_time = time.time()

        if self.algo == "ps":
            run_ps(self, grad)
        elif self.algo == "ring":
            run_ring(self, grad)
        elif self.algo == "optireduce":
            run_optireduce(self, grad)

        # ---- finish ----
        if self.node_id != 0:
            self.send(self.peers[0][0], self.peers[0][1], {"type": "DONE"})
        else:
            with self.lock:
                self.done_count += 1

            while self.done_count < self.num_nodes:
                time.sleep(0.01)

            print(f"[FINAL] {self.algo} latency = {time.time() - self.start_time:.4f}s")


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) < 2:
        print("Usage: python node.py [tcp|udp]")
        return

    mode = sys.argv[1]
    if mode not in ["tcp", "udp"]:
        print("Mode must be 'tcp' or 'udp'")
        return

    node = Node(mode)

    # start servers
    node.start_server(PORT)

    # register to coordinator
    time.sleep(1)
    node.register()

    print(f"[Node] Running in {mode.upper()} mode")

    # keep alive
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()