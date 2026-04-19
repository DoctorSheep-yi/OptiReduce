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

        self.ring_buffer = []
        self.opti_buffer = []
        self.received = []
        self.chunk_buffer = {}

    def send(self, ip, port, msg, force_mode=None):
        mode = force_mode if force_mode else self.mode
        if mode == "udp" and self.noise.should_drop_udp():
            return

        if mode == "tcp":
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((ip, port))
                sock.sendall(pickle.dumps(msg))
                sock.close()
            except: pass
        else:
            self._send_chunked(ip, port, msg)

    def _send_chunked(self, ip, port, msg):
        data = pickle.dumps(msg)
        chunk_id = str(uuid.uuid4())
        chunk_size = 800
        total = (len(data) + chunk_size - 1) // chunk_size
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(total):
            chunk = data[i * chunk_size:(i + 1) * chunk_size]
            c_msg = {"chunked": True, "chunk_id": chunk_id, "chunk_idx": i, "num_chunks": total, "payload": chunk}
            sock.sendto(pickle.dumps(c_msg), (ip, port))
            time.sleep(0.0001)
        sock.close()

    def broadcast(self, msg):
        for i, (ip, port) in enumerate(self.peers):
            if i != self.node_id: self.send(ip, port, msg, force_mode=None)

    def handle_message(self, msg):
        if msg.get("chunked"):
            cid = msg["chunk_id"]
            if cid not in self.chunk_buffer: self.chunk_buffer[cid] = {}
            self.chunk_buffer[cid][msg["chunk_idx"]] = msg["payload"]
            if len(self.chunk_buffer[cid]) == msg["num_chunks"]:
                full = b"".join([self.chunk_buffer[cid][i] for i in range(msg["num_chunks"])])
                self.handle_message(pickle.loads(full))
                del self.chunk_buffer[cid]
            return

        t = msg.get("type")
        if t == "PEER_UPDATE":
            self.peers, self.node_id, self.num_nodes = msg["peers"], msg["node_id"], len(msg["peers"])
            print(f"\n[Node {self.node_id}] Peers updated. {self.num_nodes} nodes in ring.")
        elif t == "START":
            # IMPORTANT: Start in thread so Node 0 doesn't block its own receiver
            threading.Thread(target=self.run_experiment, args=(msg,), daemon=True).start()
        elif t == "DONE":
            with self.lock: self.done_count += 1
        elif t == "NOISE":
            if msg["action"] == "straggler":
                self.noise.enable_straggler = msg["enable"]
                self.noise.sleep_time = float(msg["val"])
            elif msg["action"] == "loss":
                if msg["enable"]: self.noise.apply_packet_loss_tc(msg["val"])
                else: self.noise.clear_tc()
        elif msg.get("algo") == "ring":
            with self.lock: self.ring_buffer.append(msg)
        elif msg.get("algo") == "optireduce":
            with self.lock: self.opti_buffer.append(msg)
        elif msg.get("algo") == "ps":
            if msg.get("phase") == "push":
                with self.lock: self.received.append(msg["payload"])

    def run_experiment(self, msg):
        self.algo, size = msg["algo"], msg["size"]
        print(f"[Node {self.node_id}] STARTING {self.algo} (size={size})")
        
        self.local_matrix = generate_matrix(size)
        grad = np.tanh(self.local_matrix)
        
        self.noise.apply_straggler()
        if self.node_id == 0:
            self.done_count = 0
            self.start_time = time.perf_counter()

        if self.algo == "ps": run_ps(self, grad)
        elif self.algo == "ring": run_ring(self, grad)
        elif self.algo == "optireduce": run_optireduce(self, grad)

        if self.node_id != 0:
            self.send(self.peers[0][0], self.peers[0][1], {"type": "DONE"}, force_mode=None)
        else:
            with self.lock: self.done_count += 1
            while self.done_count < self.num_nodes: time.sleep(0.01)
            print(f"\n[FINAL] {self.algo} latency = {(time.perf_counter()-self.start_time)*1000:.2f} ms\n>> ", end="")

    def cli(self):
        print("type 'register', then 'start <algo> <size>'")
        while True:
            cmd = input(">> ").strip().split()
            if not cmd: continue
            if cmd[0] == "register":
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((COORDINATOR_IP, COORDINATOR_PORT))
                s.sendall(pickle.dumps({"type": "REGISTER", "port": PORT}))
                s.close()
            elif cmd[0] == "start":
                m = {"type": "START", "algo": cmd[1], "size": int(cmd[2])}
                self.broadcast(m)
                self.run_experiment(m)
            elif cmd[0] == "noise":
                enable = cmd[2] != "0"
                m = {"type": "NOISE", "action": cmd[1], "enable": enable, "val": cmd[2]}
                self.broadcast(m)
                self.handle_message(m)

    def start_server(self):
        def tcp():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("0.0.0.0", PORT)); s.listen(5)
            while True:
                c, _ = s.accept(); d = b""
                while True:
                    p = c.recv(65536)
                    if not p: break
                    d += p
                if d: self.handle_message(pickle.loads(d))
        def udp():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", PORT))
            while True:
                d, _ = s.recvfrom(65536)
                try: self.handle_message(pickle.loads(d))
                except: pass
        threading.Thread(target=tcp, daemon=True).start()
        threading.Thread(target=udp, daemon=True).start()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["tcp", "udp"], required=True)
    a = p.parse_args()
    n = Node(a.mode)
    n.start_server()
    n.cli()