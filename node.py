import socket
import argparse
import threading
import struct
import pickle
import random
import sys

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

        self.lock = threading.Lock()

    # -------------------------
    # REGISTER
    # -------------------------
    def register(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((COORDINATOR_IP, COORDINATOR_PORT))

        msg = {
            "type": "REGISTER",
            "port": PORT
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

    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", PORT))
        server.listen()

        print("[TCP] Listening...")

        while True:
            conn, addr = server.accept()
            threading.Thread(target=self.handle_tcp, args=(conn,), daemon=True).start()

    def handle_tcp(self, conn):
        data = conn.recv(4096)

        # Try coordinator message
        try:
            msg = pickle.loads(data)

            if msg["type"] == "PEER_UPDATE":
                self.peers = msg["peers"]
                self.node_id = msg["node_id"]
                self.num_nodes = len(self.peers)

                print(f"\n[Node] Updated peers: {self.peers}")
                print(f"My ID: {self.node_id}\n")
                return

            elif msg["type"] == "DATA":
                print(f"[RECV][TCP] From Node {msg['from']} → {msg['value']}")
                return

        except:
            pass

        conn.close()

    def udp_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", PORT))

        print("[UDP] Listening...")

        while True:
            data, addr = sock.recvfrom(4096)

            try:
                msg = pickle.loads(data)

                if msg["type"] == "PEER_UPDATE":
                    self.peers = msg["peers"]
                    self.node_id = msg["node_id"]
                    self.num_nodes = len(self.peers)

                    print(f"\n[Node] Updated peers: {self.peers}")
                    print(f"My ID: {self.node_id}\n")
                    continue

                elif msg["type"] == "DATA":
                    print(f"[RECV][UDP] From Node {msg['from']} → {msg['value']}")
                    continue

            except:
                pass

    # -------------------------
    # SEND RANDOM NUMBER
    # -------------------------
    def send_random(self, target_id):
        if target_id >= self.num_nodes:
            print("Invalid target node")
            return

        ip, port = self.peers[target_id]

        value = random.randint(0, 100)

        msg = {
            "type": "DATA",
            "from": self.node_id,
            "value": value
        }

        payload = pickle.dumps(msg)

        print(f"[SEND] → Node {target_id} ({ip}) : {value}")

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
        print("send <node_id>")
        print("quit")

        while True:
            cmd = input("> ").strip()

            if cmd == "register":
                self.register()

            elif cmd.startswith("send"):
                if self.num_nodes is None:
                    print("Not registered yet!")
                    continue

                parts = cmd.split()
                if len(parts) != 2:
                    print("Usage: send <node_id>")
                    continue

                target = int(parts[1])
                self.send_random(target)

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