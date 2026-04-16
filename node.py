import socket
import threading
import pickle
import time
import numpy as np
import uuid

from matrix import generate_matrix
from PS import run_ps
from ring_allreduce import run_ring
from optireduce import run_optireduce
from noise import Noise
from globals import *

COORDINATOR_IP = "172.21.102.115"
COORDINATOR_PORT = 8000
PORT = 9000

class Node:
    def __init__(self, mode):
        self.mode = mode # 'tcp' or 'udp'
        self.peers = []
        self.node_id = None
        self.num_nodes = None

        self.local_matrix = None
        self.done_count = 0
        self.lock = threading.Lock()

        self.algo = None
        self.start_time = None
        self.noise = Noise()

        # Buffers for algorithms
        self.ring_buffer = []
        self.opti_buffer = []
        self.received = [] # For PS server
        self.chunk_buffer = {}

    def send(self, ip, port, msg, force_mode=None):
        """Sends data. force_mode allows overriding the global mode (e.g., PS always needs reliability)."""
        mode = force_mode if force_mode else self.mode
        data = pickle.dumps(msg)

        if mode == "tcp":
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((ip, port))
                sock.sendall(data)
                sock.close()
            except Exception as e:
                print(f"[TCP Send Error] {e}")
        else:
            # UDP path with basic chunking for large gradients
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if len(data) <= MAX_CHUNK_SIZE:
                sock.sendto(data, (ip, port))
            else:
                chunk_id = str(uuid.uuid4())[:8]
                chunk_size = 1024
                total = (len(data) + chunk_size - 1) // chunk_size
                for i in range(total):
                    chunk = data[i*chunk_size : (i+1)*chunk_size]
                    c_msg = {"chunked":True, "cid":chunk_id, "idx":i, "tot":total, "pay":chunk}
                    sock.sendto(pickle.dumps(c_msg), (ip, port))
                    time.sleep(0.0001) # Simple rate control
            sock.close()

    def broadcast(self, msg):
        for i, (ip, port) in enumerate(self.peers):
            if i != self.node_id:
                self.send(ip, port, msg, force_mode="tcp")

    def start_server(self):
        threading.Thread(target=self.tcp_server, daemon=True).start()
        threading.Thread(target=self.udp_server, daemon=True).start()

    def tcp_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("0.0.0.0", PORT))
        s.listen(5)
        while True:
            conn, _ = s.accept()
            data = b""
            while True:
                packet = conn.recv(65536)
                if not packet: break
                data += packet
            if data:
                try: self.handle_message(pickle.loads(data))
                except: pass
            conn.close()

    def udp_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", PORT))
        while True:
            data, _ = s.recvfrom(65536)
            try:
                m = pickle.loads(data)
                if m.get("chunked"):
                    cid = m["cid"]
                    if cid not in self.chunk_buffer: self.chunk_buffer[cid] = {}
                    self.chunk_buffer[cid][m["idx"]] = m["pay"]
                    if len(self.chunk_buffer[cid]) == m["tot"]:
                        full_data = b"".join([self.chunk_buffer[cid][i] for i in range(m["tot"])])
                        self.handle_message(pickle.loads(full_data))
                        del self.chunk_buffer[cid]
                else:
                    self.handle_message(m)
            except: pass

    def handle_message(self, msg):
        t = msg.get("type")
        if t == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(self.peers)
        elif t == "START":
            threading.Thread(target=self.run_experiment, args=(msg,), daemon=True).start()
        elif t == "DONE":
            with self.lock: self.done_count += 1
        elif t == "NOISE":
            action = msg["action"]
            if action == "straggler":
                self.noise.enable_straggler = msg["enable"]
                self.noise.sleep_time = float(msg["val"])
                print(f"[Node {self.node_id}] Straggler Noise: {msg['enable']} ({msg['val']}s)")
                
            elif action == "loss":
                if msg["enable"]:
                    # Applies to the whole OS network stack
                    self.noise.apply_packet_loss_tc(msg["val"])
                else:
                    self.noise.clear_tc()
        elif msg.get("algo") == "ring":
            with self.lock: self.ring_buffer.append(msg)
        elif msg.get("algo") == "optireduce":
            with self.lock: self.opti_buffer.append(msg)
        elif msg.get("algo") == "ps":
            if msg.get("phase") == "push":
                with self.lock: self.received.append(msg["payload"])

    def run_experiment(self, msg):
        self.algo = msg["algo"]
        size = msg["size"]
        self.local_matrix = generate_matrix(size)
        gradient = np.tanh(self.local_matrix)
        
        # Apply straggler variability
        self.noise.apply_straggler()

        if self.node_id == 0:
            self.done_count = 0
            self.start_time = time.perf_counter()

        # Dispatch
        if self.algo == "ps": result = run_ps(self, gradient)
        elif self.algo == "ring": result = run_ring(self, gradient)
        elif self.algo == "optireduce": result = run_optireduce(self, gradient)

        # Sync Finish
        if self.node_id != 0:
            self.send(self.peers[0][0], self.peers[0][1], {"type": "DONE"}, force_mode="tcp")
        else:
            with self.lock: self.done_count += 1
            while self.done_count < self.num_nodes: time.sleep(0.01)
            print(f"\n[RESULT] {self.algo} Latency: {(time.perf_counter()-self.start_time)*1000:.2f} ms\n>> ", end="")

    def register(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((COORDINATOR_IP, COORDINATOR_PORT))
        s.sendall(pickle.dumps({"type": "REGISTER", "port": PORT}))
        s.close()

    def cli(self):
        while True:
            cmd = input(">> ").strip().split()
            if not cmd: continue
            if cmd[0] == "start":
                self.broadcast({"type": "START", "algo": cmd[1], "size": int(cmd[2])})
                self.run_experiment({"type": "START", "algo": cmd[1], "size": int(cmd[2])})
            elif cmd[0] == "noise":
                enable = cmd[2] != "0"
                msg = {"type":"NOISE", "action":cmd[1], "enable":enable, "val":cmd[2]}
                self.broadcast(msg)
                self.handle_message(msg)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["tcp", "udp"], default="tcp")
    a = p.parse_args()
    n = Node(a.mode)
    n.start_server()
    print("Type 'reg' to register")
    if input("> ") == "reg": n.register()
    n.cli()