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
# TCP helpers
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
        self.node_matrix_size = None

        self.lock = threading.Lock()
        self.done_count = 0

        self.noise = Noise()

        self.ring_buffer = []
        self.opti_buffer = []
        self.received = []
        self.chunk_buffer = {}
        self.report_buffer = []

        self.start_time = None

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

                # ✅ increase buffer (important for large matrices)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_BUFFER_SIZE)

                sock.connect((ip, port))
                sock.sendall(len(data).to_bytes(4, 'big') + data)
                sock.close()
            except Exception as e:
                print(f"[TCP SEND ERROR] {e}")
        else:
            self._send_chunked(ip, port, data)


    def _send_chunked(self, ip, port, data):
        chunk_id = str(uuid.uuid4())
        size = MAX_CHUNK_SIZE
        total = (len(data) + size - 1) // size

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except:
            pass

        for i in range(total):
            chunk = data[i * size:(i + 1) * size]
            msg = {
                "chunked": True,
                "chunk_id": chunk_id,
                "chunk_idx": i,
                "num_chunks": total,
                "payload": chunk
            }

            payload = pickle.dumps(msg)

            
            while True:
                try:
                    sock.sendto(payload, (ip, port))
                    break
                except OSError as e:
                    if e.errno == 55:  # No buffer space
                        time.sleep(0.001)  
                    else:
                        raise

            
            time.sleep(0.0005)

        sock.close()

    # =========================
    # RECEIVE
    # =========================
    def handle_message(self, msg):
        # UDP reassembly
        if msg.get("chunked"):
            cid = msg["chunk_id"]
            if cid not in self.chunk_buffer:
                self.chunk_buffer[cid] = {}

            self.chunk_buffer[cid][msg["chunk_idx"]] = msg["payload"]

            if len(self.chunk_buffer[cid]) == msg["num_chunks"]:
                full = b''.join(self.chunk_buffer[cid][i] for i in range(msg["num_chunks"]))
                del self.chunk_buffer[cid]
                self.handle_message(pickle.loads(full))
            return

        t = msg.get("type")

        # ===== PEER UPDATE =====
        if t == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(msg["peers"])

            print("\n" + "="*50)
            print(f"[Node {self.node_id}] Joined cluster")
            print(f"[Node {self.node_id}] Total nodes: {self.num_nodes}")
            print(f"[Node {self.node_id}] Peers:")
            for i, (ip, port) in enumerate(self.peers):
                tag = " (ME)" if i == self.node_id else ""
                print(f"   - Node {i}: {ip}:{port}{tag}")
            print("="*50 + "\n")

        elif t == "START":
            threading.Thread(target=self.run_experiment, args=(msg,), daemon=True).start()
        
        elif t == "REPORT":
            # Node 0 collects these to calculate global accuracy
            with self.lock:
                self.report_buffer.append(msg)

        elif t == "DONE":
            with self.lock:
                self.done_count += 1

        # ===== PS =====
        elif msg.get("algo") == "ps":
            if msg.get("phase") == "push":
                with self.lock:
                    self.received.append(msg["payload"])

            elif msg.get("phase") == "pop":
                print("[PS] Received results")

        # ===== RING =====
        elif msg.get("algo") == "ring":
            with self.lock:
                self.ring_buffer.append(msg)

        # ===== OPTIREDUCE =====
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

        while True:
            conn, _ = server.accept()
            try:
                length_bytes = recv_exact(conn, 4)
                if not length_bytes:
                    continue

                length = int.from_bytes(length_bytes, 'big')
                if 0 < length < 1_000_000_000:
                    data = recv_exact(conn, length)
                    if data:
                        msg = pickle.loads(data)
                        self.handle_message(msg)
            except Exception as e:
                print(f"[TCP RECV ERROR] {e}")
            finally:
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
    # COORDINATOR
    # =========================
    def register(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((COORDINATOR_IP, COORDINATOR_PORT))

            msg = {"type": "REGISTER", "port": PORT}
            data = pickle.dumps(msg)
            # ✅ Consistency: Prepend length even for registration
            sock.sendall(len(data).to_bytes(4, 'big') + data)
            sock.close()
        except Exception as e:
            print(f"[Register Error] {e}")

    # =========================
    # EXPERIMENT
    # =========================
    def clear_buffers(self):
        """Wipes old data so experiments don't mix."""
        with self.lock:
            self.ring_buffer = []
            self.opti_buffer = []
            self.received = []
            self.report_buffer = []
            self.chunk_buffer = {}

    def run_experiment(self, msg):
        algo = msg["algo"]
        size = msg["size"]

        print(f"\n[Node {self.node_id}] ===== START {algo.upper()} =====")
        print(f"[Node {self.node_id}] Matrix size: {size}")

        self.clear_buffers()
        grad = generate_matrix(size)
        self.node_matrix_size = size
        self.start_time = time.time()

        if self.node_id == 0:
            self.done_count = 0

        if algo == "ps":
            run_ps(self, grad)

        elif algo == "ring":
            run_ring(self, grad)

        elif algo == "optireduce":
            approx_res, original = run_optireduce(self, grad)
            latency = (time.time() - self.start_time) * 1000

            # # Send report to Node 0 (reliable)
            # report = {
            #     "type": "REPORT",
            #     "node_id": self.node_id,
            #     "approx": approx_res,
            #     "truth": original
            # }

            # self.send(self.peers[0][0], self.peers[0][1], report, force_mode="tcp")

            # # Node 0 computes accuracy
            # if self.node_id == 0:
            #     self.calculate_accuracy(latency, size)

        if self.node_id == 0:
            print("\n" + "=" * 50)
            print(f"[RESULT] Algorithm: {algo}")
            latency_ms = (time.time() - self.start_time) * 1000
            print(f"[RESULT] Latency: {latency_ms:.2f} ms")
            print("=" * 50 + "\n")


    def calculate_accuracy(self, latency, size):
        # Wait for all reports
        while True:
            with self.lock:
                if len(self.report_buffer) >= self.num_nodes:
                    reports = list(self.report_buffer)
                    break
            time.sleep(0.01)

        # ===== Ground truth (global average) =====
        true_avg = sum(r["truth"] for r in reports) / len(reports)

        # ===== Approximation (average across nodes) =====
        approx_avg = sum(r["approx"] for r in reports) / len(reports)

        # ===== Metrics =====
        mse = np.mean((true_avg - approx_avg) ** 2)
        mae = np.mean(np.abs(true_avg - approx_avg))

        # ===== Accuracy (%) =====
        numerator = np.linalg.norm(true_avg - approx_avg)
        denominator = np.linalg.norm(true_avg) + 1e-12
        accuracy = (1 - numerator / denominator) * 100
        accuracy = max(0.0, accuracy)

        print("\n" + "=" * 50)
        print(f"ALGORITHM: OPTIREDUCE | MATRIX SIZE: {size}")
        print(f"LATENCY: {latency:.2f} ms")
        print(f"MSE: {mse:.8e}")
        print(f"MAE: {mae:.8e}")
        print(f"ACCURACY: {accuracy:.2f}%")
        print("=" * 50 + "\n")



    # =========================
    # CLI
    # =========================
    def cli(self):
        while True:
            try:
                cmd = input("\nCommand (start [ps|ring|optireduce] size): ").strip()
                parts = cmd.split()

                if len(parts) == 3 and parts[0] == "start":
                    if self.node_id != 0:
                        print("[ERROR] Only Node 0 can start")
                        continue

                    algo = parts[1]
                    size = int(parts[2])

                    msg = {"type": "START", "algo": algo, "size": size}

                    print(f"[Node 0] Broadcasting {algo}")

                    for ip, port in self.peers:
                        self.send(ip, port, msg)

                else:
                    print("Invalid command")

            except Exception as e:
                print(f"[CLI ERROR] {e}")


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) < 2:
        print("Usage: python node.py [tcp|udp]")
        return

    mode = sys.argv[1]

    node = Node(mode)
    node.start_server(PORT)

    time.sleep(1)
    node.register()

    print(f"[Node] Running in {mode.upper()} mode")
    print("\nCommands:")
    print("  start ps <size>")
    print("  start ring <size>")
    print("  start optireduce <size>")
    print("Only Node 0 can start experiments.\n")

    node.cli()


if __name__ == "__main__":
    main()