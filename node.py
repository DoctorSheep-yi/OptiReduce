import socket
import argparse
import threading
import pickle
import uuid
import sys
import numpy as np
from matrix import generate_matrix
import PS

COORDINATOR_IP = "172.21.102.115"
COORDINATOR_PORT = 8000
PORT = 9000

UDP_SAFE = 1200
CHUNK_SIZE = 1000


class Node:
    def __init__(self, mode):
        self.mode = mode
        self.peers = []
        self.node_id = None
        self.num_nodes = None

        self.received_matrices = []
        self.chunk_buffer = {}
        self.final_result = None

    # -------------------------
    # REGISTER
    # -------------------------
    def register(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((COORDINATOR_IP, COORDINATOR_PORT))
        msg = {"type": "REGISTER", "port": PORT}
        sock.sendall(pickle.dumps(msg))
        sock.close()

    def unregister(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((COORDINATOR_IP, COORDINATOR_PORT))
            msg = {"type": "UNREGISTER", "port": PORT}
            sock.sendall(pickle.dumps(msg))
            sock.close()
            print("[Node] Unregistered")
        except Exception as e:
            print("[Node] Unregister failed:", e)

    # -------------------------
    # SERVER
    # -------------------------
    def start_server(self):
        threading.Thread(target=self.tcp_server, daemon=True).start()
        if self.mode == "udp":
            threading.Thread(target=self.udp_server, daemon=True).start()

    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", PORT))
        server.listen()
        print("[TCP] Listening...")

        while True:
            conn, _ = server.accept()
            threading.Thread(target=self.handle_tcp, args=(conn,), daemon=True).start()

    def handle_tcp(self, conn):
        try:
            data = b''
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet

            msg = pickle.loads(data)
            self.process_message(msg)

        except Exception as e:
            print("[TCP ERROR]", e)
        finally:
            conn.close()

    def udp_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", PORT))
        print("[UDP] Listening...")

        while True:
            try:
                data, _ = sock.recvfrom(65535)
                msg = pickle.loads(data)
                self.process_message(msg)
            except Exception as e:
                print("[UDP ERROR]", e)

    # -------------------------
    # MESSAGE HANDLER
    # -------------------------
    def process_message(self, msg):
        msg_type = msg.get("type")

        if msg_type == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(self.peers)

            print(f"\n[Node] Peers: {self.peers}")
            print(f"My ID: {self.node_id}\n")

        elif msg_type == "DATA":
            data = np.array(msg["data"])

            print(f"[RECV] FULL from Node {msg['from']}")
            self.received_matrices.append(data)
            print(f"[STORE] total={len(self.received_matrices)}")

        elif msg_type == "DATA_CHUNK":
            self.handle_chunk(msg)

        elif msg_type == "RESULT":
            self.final_result = np.array(msg["data"])
            print("[RESULT RECEIVED] (use 'show result')")

        elif msg_type == "RESULT_CHUNK":
            self.handle_result_chunk(msg)

    # -------------------------
    # CHUNK HANDLING (DATA)
    # -------------------------
    def handle_chunk(self, msg):
        msg_id = msg["msg_id"]
        seq = msg["seq"]
        total = msg["total"]
        chunk = msg["data"]

        if msg_id not in self.chunk_buffer:
            self.chunk_buffer[msg_id] = [None] * total

        self.chunk_buffer[msg_id][seq] = chunk

        if all(c is not None for c in self.chunk_buffer[msg_id]):
            full_payload = b''.join(self.chunk_buffer[msg_id])
            full_msg = pickle.loads(full_payload)

            data = np.array(full_msg["data"])

            self.received_matrices.append(data)

            print(f"[RECV] Matrix reconstructed")
            print(f"[STORE] total={len(self.received_matrices)}")

            del self.chunk_buffer[msg_id]

    # -------------------------
    # CHUNK HANDLING (RESULT)
    # -------------------------
    def handle_result_chunk(self, msg):
        msg_id = msg["msg_id"]
        seq = msg["seq"]
        total = msg["total"]
        chunk = msg["data"]

        if msg_id not in self.chunk_buffer:
            self.chunk_buffer[msg_id] = [None] * total

        self.chunk_buffer[msg_id][seq] = chunk

        if all(c is not None for c in self.chunk_buffer[msg_id]):
            full_payload = b''.join(self.chunk_buffer[msg_id])
            full_msg = pickle.loads(full_payload)

            self.final_result = np.array(full_msg["data"])

            print("[RESULT STORED] (use 'show result')")

            del self.chunk_buffer[msg_id]

    # -------------------------
    # SEND HELPERS
    # -------------------------
    def _send_payload(self, ip, port, payload):
        if self.mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            sock.sendall(payload)
            sock.close()
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(payload, (ip, port))

    def _send_large(self, ip, port, msg, chunk_type):
        payload = pickle.dumps(msg)

        if len(payload) <= UDP_SAFE:
            self._send_payload(ip, port, payload)
            return

        msg_id = str(uuid.uuid4())
        total = (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for i in range(total):
            chunk = payload[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]

            chunk_msg = {
                "type": chunk_type,
                "msg_id": msg_id,
                "seq": i,
                "total": total,
                "data": chunk
            }

            self._send_payload(ip, port, pickle.dumps(chunk_msg))

    # -------------------------
    # SEND MATRIX
    # -------------------------
    def send_matrix(self, target_id, data):
        ip, port = self.peers[target_id]

        msg = {
            "type": "DATA",
            "from": self.node_id,
            "data": data.tolist()
        }

        print(f"[SEND] → Node {target_id}")
        self._send_large(ip, port, msg, "DATA_CHUNK")

    # -------------------------
    # BROADCAST RESULT
    # -------------------------
    def broadcast_result(self, result):
        msg = {
            "type": "RESULT",
            "from": self.node_id,
            "data": result.tolist()
        }

        for i, (ip, port) in enumerate(self.peers):
            print(f"[BROADCAST] → Node {i}")
            self._send_large(ip, port, msg, "RESULT_CHUNK")

    # -------------------------
    # RUN
    # -------------------------
    def run(self):
        self.start_server()

        matrix = None

        print("Commands:")
        print("register")
        print("generate matrix <size>")
        print("send <node_id>")
        print("sum")
        print("multiply")
        print("show result")
        print("quit")

        while True:
            cmd = input("> ").strip()

            if cmd == "register":
                self.register()

            elif cmd.startswith("generate matrix"):
                size = int(cmd.split()[2])
                matrix = generate_matrix(size)
                print(f"Generated {matrix.shape}")

            elif cmd.startswith("send"):
                target = int(cmd.split()[1])
                self.send_matrix(target, matrix)

            elif cmd == "sum":
                if not self.received_matrices:
                    print("No matrices")
                    continue

                result = PS.sum(self.received_matrices)
                print("[SUM DONE]")
                self.broadcast_result(result)

            elif cmd == "multiply":
                if not self.received_matrices:
                    print("No matrices")
                    continue

                result = PS.multiply(self.received_matrices)
                print("[MULTIPLY DONE]")
                self.broadcast_result(result)

            elif cmd == "show result":
                if self.final_result is None:
                    print("No result available")
                else:
                    print("[RESULT]")
                    print(self.final_result)
                    print("shape:", self.final_result.shape)

            elif cmd == "quit":
                self.unregister()
                sys.exit(0)

            else:
                print("Unknown command")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tcp", "udp"], required=True)
    args = parser.parse_args()

    node = Node(args.mode)
    node.run()