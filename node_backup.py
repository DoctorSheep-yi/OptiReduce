import socket
import argparse
import threading
import struct
import pickle
import random
import sys
from matrix import generate_matrix
import PS

COORDINATOR_IP = "172.21.102.115"
COORDINATOR_PORT = 8000

PORT = 9000


# =========================
# NODE
# =========================

class Node:
    def __init__(self, mode):
        self.mode = mode
        self.peers = []
        self.node_id = None
        self.num_nodes = None

        # shard_id -> list of received shards
        self.shard_buffer = {}

        self.lock = threading.Lock()

    # -------------------------
    # REGISTER
    # -------------------------
    def register(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((COORDINATOR_IP, COORDINATOR_PORT))

        msg = {
            "type": "REGISTER",
            "port": COORDINATOR_PORT
        }

        sock.sendall(pickle.dumps(msg))
        sock.close()

    # -------------------------
    # SERVER
    # -------------------------
    def start_server(self):
        if self.mode == "tcp":
            threading.Thread(target=self.tcp_server, daemon=True).start()
        else:
            threading.Thread(target=self.udp_server, daemon=True).start()

    # ---------------------------
    # TCP SERVER
    # ---------------------------
    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", PORT))
        server.listen()

        print("[TCP] Listening...")

        while True:
            conn, addr = server.accept()
            threading.Thread(
                target=self.handle_tcp, args=(conn,), daemon=True
            ).start()

    def handle_tcp(self, conn):
        try:
            # 🔹 receive full message (IMPORTANT)
            data = b''
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet

            msg = pickle.loads(data)

            self.process_message(msg, protocol="TCP")

        except Exception as e:
            print("[TCP ERROR]", e)

        finally:
            conn.close()

    # ---------------------------
    # UDP SERVER
    # ---------------------------
    def udp_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", PORT))

        print("[UDP] Listening...")

        while True:
            try:
                data, addr = sock.recvfrom(65535)
                msg = pickle.loads(data)
                self.process_message(msg, protocol="UDP")

            except Exception as e:
                print("[UDP ERROR]", e)

    # ---------------------------
    # MESSAGE PROCESSING
    # ---------------------------
    def process_message(self, msg, protocol="TCP"):
        msg_type = msg.get("type")

        if msg_type == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(self.peers)

            print(f"\n[Node] Updated peers: {self.peers}")
            print(f"My ID: {self.node_id}\n")

        elif msg_type == "DATA":
            self.handle_data(msg, protocol)

    # ---------------------------
    # HANDLE SHARD DATA
    # ---------------------------
    def handle_data(self, msg, protocol):
        sender = msg["from"]
        shard_id = msg["shard_id"]
        shard = msg["data"]

        print(f"[RECV][{protocol}] From Node {sender} → shard {shard_id}")

        with self.lock:
            if shard_id not in self.shard_buffer:
                self.shard_buffer[shard_id] = []

            self.shard_buffer[shard_id].append(shard)

            # 🔹 check if all shards received
            if len(self.shard_buffer[shard_id]) == self.num_nodes:
                print(f"[AGGREGATE] Shard {shard_id} ready")

                result = PS.combine_results(
                    self.shard_buffer[shard_id], method="sum"
                )

                print(
                    f"[RESULT] Shard {shard_id} aggregated "
                    f"shape: {getattr(result, 'shape', len(result))}"
                )

                # clear buffer for next round
                self.shard_buffer[shard_id] = []

                #TODO: broadcast result to all nodes
                # self.broadcast_result(shard_id, result)

    # -------------------------
    # SEND SHARD
    # -------------------------
    def send_shard(self, target_id, data):
        if target_id >= self.num_nodes:
            print("Invalid target node")
            return

        ip, port = self.peers[target_id]

        # 🔹 create shards
        shards = PS.shard_data(data, self.num_nodes)

        # 🔹 pick the shard for this target
        shard = shards[target_id]

        msg = {
            "type": "DATA",
            "from": self.node_id,
            "shard_id": target_id,   # important for server logic
            "data": shard
        }

        payload = pickle.dumps(msg)

        print(f"[SEND] → Node {target_id} shard shape: {getattr(shard, 'shape', len(shard))}")

        if self.mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            sock.sendall(payload)
            sock.close()
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(payload, (ip, port))
    # -------------------------
    # SEND ALL SHARDs
    # -------------------------
    def send_all_shards(self, data):
        shards = PS.shard_data(data, self.num_nodes)

        for target_id in range(self.num_nodes):
            ip, port = self.peers[target_id]
            shard = shards[target_id]

            msg = {
                "type": "DATA",
                "from": self.node_id,
                "shard_id": target_id,
                "data": shard
            }

            payload = pickle.dumps(msg)

            print(f"[SEND] → Node {target_id} shard shape: {getattr(shard, 'shape', len(shard))}")

            if self.mode == "tcp":
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((ip, port))
                sock.sendall(payload)
                sock.close()
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(payload, (ip, port))
    # -------------------------
    # RUN
    # -------------------------
    def run(self):
        self.start_server()

        print("Commands:")
        print("register")
        print("generate matrix <size>")
        print("send <node_id>")
        print("send_all")
        print("quit")

        while True:
            cmd = input("> ").strip()

            if cmd == "register":
                self.register()
            elif cmd.startswith("generate matrix"):
                parts = cmd.split()
                if len(parts) != 3:
                    print("Usage: generate matrix <size>")
                    continue

                size = int(parts[2])
                matrix = generate_matrix(size)
                print(f"Generated matrix of shape {matrix.shape}")
            elif cmd.startswith("send") and cmd != "send_all":
                if self.num_nodes is None:
                    print("Not registered yet!")
                    continue

                parts = cmd.split()
                if len(parts) != 2:
                    print("Usage: send <node_id>")
                    continue

                target = int(parts[1])
                self.send_shard(target, matrix)

            elif cmd == "send_all":
                if self.num_nodes is None:
                    print("Not registered yet!")
                    continue

                self.send_all_shards(matrix)

            elif cmd == "quit":
                print("Shutting down node...")
                sys.exit(0)

            else:
                print("Unknown command")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tcp", "udp"], required=True)

    args = parser.parse_args()

    node = Node(args.mode)
    node.run()