import socket
import argparse
import threading
import pickle
import sys
import time

import numpy as np

from matrix import generate_matrix
from node import Node, PORT

np.set_printoptions(precision=3, suppress=True)


class OptiReduceNode(Node):
    def __init__(self, mode):
        super().__init__(mode)
        self.local_matrix = None
        self.tb = None
        self.jobs = {}
        self.results = {}

    def start_server(self):
        threading.Thread(target=self.tcp_server, daemon=True).start()
        threading.Thread(target=self.udp_server, daemon=True).start()

    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", PORT))
        server.listen()
        print("[TCP] Listening...")

        while True:
            conn, _ = server.accept()
            threading.Thread(target=self.handle_tcp, args=(conn,), daemon=True).start()

    def handle_tcp(self, conn):
        pieces = []
        try:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                pieces.append(data)

            if pieces:
                msg = pickle.loads(b"".join(pieces))
                self.handle_message(msg)
        except Exception as e:
            print("[TCP] Error:", e)
        finally:
            conn.close()

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
                print("[UDP] Error:", e)

    def handle_message(self, msg):
        msg_type = msg.get("type")

        if msg_type == "PEER_UPDATE":
            self.peers = msg["peers"]
            self.node_id = msg["node_id"]
            self.num_nodes = len(self.peers)
            print(f"\n[Node] Updated peers: {self.peers}")
            print(f"My ID: {self.node_id}\n")
            return

        if msg_type == "DATA":
            print(f"[RECV][{msg.get('transport', 'GEN')}] From Node {msg['from']} -> {msg['value']}")
            return

        if msg_type == "WARMUP_PING":
            self.reply_warmup(msg)
            return

        if msg_type == "WARMUP_ACK":
            self.got_warmup_ack(msg)
            return

        if msg_type == "OPTI_START":
            self.start_round(msg)
            return

        if msg_type == "OPTI_SHARD":
            self.store_shard(msg)
            return

        if msg_type == "OPTI_AGG":
            self.store_final_shard(msg)
            return

    def send_msg(self, ip, port, msg, mode):
        payload = pickle.dumps(msg)

        if mode == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            sock.sendall(payload)
            sock.close()
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(payload, (ip, port))
            sock.close()

    def make_matrix(self, size):
        self.local_matrix = generate_matrix(size).astype(float)
        print(f"[Matrix] Node {self.node_id} local matrix:")
        print(self.local_matrix)

    def show_matrix(self):
        if self.local_matrix is None:
            print("[Matrix] No local matrix yet")
            return

        print(f"[Matrix] Node {self.node_id} local matrix:")
        print(self.local_matrix)

    def ensure_matrix(self, size):
        if self.local_matrix is None or self.local_matrix.shape != (size, size):
            self.local_matrix = generate_matrix(size).astype(float)

    def split_rows(self, size):
        parts = []
        start = 0

        for chunk in np.array_split(np.arange(size), self.num_nodes):
            end = start + len(chunk)
            parts.append((start, end))
            start = end

        return parts

    def build_hadamard(self, n):
        h = np.array([[1.0]])
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        return h

    def encode_shard(self, shard):
        shard = np.asarray(shard, dtype=float)
        cols = shard.shape[1]
        size = 1 if cols <= 1 else 1 << (cols - 1).bit_length()

        if size != cols:
            shard = np.pad(shard, ((0, 0), (0, size - cols)), mode="constant")

        h = self.build_hadamard(size)
        encoded = shard @ h / np.sqrt(size)
        return encoded, cols

    def decode_shard(self, encoded, original_cols):
        encoded = np.asarray(encoded, dtype=float)
        h = self.build_hadamard(encoded.shape[1])
        decoded = encoded @ h / np.sqrt(encoded.shape[1])
        return decoded[:, :original_cols]

    def make_job(self, job_id, timeout, row_parts, size):
        with self.lock:
            self.jobs[job_id] = {
                "timeout": timeout,
                "row_parts": row_parts,
                "size": size,
                "incoming": {},
                "acks": 0,
                "ack_event": threading.Event(),
            }

    def reply_warmup(self, msg):
        ip, port = self.peers[msg["source_id"]]
        reply = {
            "type": "WARMUP_ACK",
            "job_id": msg["job_id"],
            "from": self.node_id,
        }
        self.send_msg(ip, port, reply, "tcp")

    def got_warmup_ack(self, msg):
        with self.lock:
            job = self.jobs.get(msg["job_id"])
            if job is None:
                return
            job["acks"] += 1
            job["ack_event"].set()

    def calibrate_timeout(self, warmup_rounds=5):
        if self.num_nodes is None:
            print("Not registered yet!")
            return

        times = []

        for i in range(warmup_rounds):
            job_id = f"warmup-{self.node_id}-{i}-{int(time.time() * 1000)}"
            self.make_job(job_id, None, [], 0)

            start = time.time()

            for peer_id, (ip, port) in enumerate(self.peers):
                if peer_id == self.node_id:
                    continue

                msg = {
                    "type": "WARMUP_PING",
                    "job_id": job_id,
                    "source_id": self.node_id,
                }
                self.send_msg(ip, port, msg, "tcp")

            while True:
                with self.lock:
                    done = self.jobs[job_id]["acks"] >= self.num_nodes - 1
                if done:
                    break

                self.jobs[job_id]["ack_event"].wait(timeout=1)
                self.jobs[job_id]["ack_event"].clear()

            elapsed = time.time() - start
            times.append(elapsed)

            with self.lock:
                del self.jobs[job_id]

            print(f"[Warmup] Round {i + 1}: {elapsed:.4f}s")

        self.tb = float(np.percentile(times, 95))
        print(f"[Warmup] tB = {self.tb:.4f}s")

    def start_round(self, msg):
        size = msg["size"]
        job_id = msg["job_id"]
        timeout = msg["timeout"]
        row_parts = msg["row_ranges"]
        straggler_id = msg.get("straggler_id")
        delay = msg.get("delay", 0.0)

        self.ensure_matrix(size)
        self.make_job(job_id, timeout, row_parts, size)

        if self.node_id == straggler_id and delay > 0:
            print(f"[Delay] Node {self.node_id} sleeping for {delay}s")
            time.sleep(delay)

        for owner_id, (start_row, end_row) in enumerate(row_parts):
            shard = self.local_matrix[start_row:end_row]
            encoded, original_cols = self.encode_shard(shard)

            msg_out = {
                "type": "OPTI_SHARD",
                "job_id": job_id,
                "owner_id": owner_id,
                "source_id": self.node_id,
                "start_row": start_row,
                "end_row": end_row,
                "encoded_shard": encoded.tolist(),
                "original_cols": original_cols,
            }

            ip, port = self.peers[owner_id]
            if owner_id == self.node_id:
                self.store_shard(msg_out)
            else:
                self.send_msg(ip, port, msg_out, "udp")

        threading.Thread(target=self.aggregate_shards, args=(job_id,), daemon=True).start()

    def store_shard(self, msg):
        with self.lock:
            job = self.jobs.get(msg["job_id"])
            if job is None:
                return

            bucket = job["incoming"].setdefault(
                msg["start_row"],
                {"encoded": [], "original_cols": msg["original_cols"]},
            )
            bucket["encoded"].append(np.array(msg["encoded_shard"], dtype=float))

    def aggregate_shards(self, job_id):
        with self.lock:
            row_parts = self.jobs[job_id]["row_parts"]
            timeout = self.jobs[job_id]["timeout"]

        for owner_id, (start_row, end_row) in enumerate(row_parts):
            if owner_id != self.node_id:
                continue

            start = time.time()
            while time.time() - start < timeout:
                time.sleep(0.02)

            with self.lock:
                job = self.jobs[job_id]
                bucket = job["incoming"].get(start_row, {"encoded": []})
                size = job["size"]

            count = len(bucket["encoded"])

            if count == 0:
                final_shard = np.zeros((end_row - start_row, size), dtype=float)
            else:
                total = np.sum(bucket["encoded"], axis=0)
                avg = total / count
                final_shard = self.decode_shard(avg, bucket["original_cols"])

            print(f"[Shard] Node {self.node_id} aggregated rows {start_row}-{end_row - 1} from {count} shard(s)")
            self.broadcast_shard(job_id, start_row, final_shard)

    def broadcast_shard(self, job_id, start_row, final_shard):
        msg = {
            "type": "OPTI_AGG",
            "job_id": job_id,
            "start_row": start_row,
            "agg_shard": final_shard.tolist(),
            "owner_id": self.node_id,
        }

        for peer_id, (ip, port) in enumerate(self.peers):
            if peer_id == self.node_id:
                self.store_final_shard(msg)
            else:
                self.send_msg(ip, port, msg, "udp")

    def store_final_shard(self, msg):
        with self.lock:
            bucket = self.results.setdefault(
                msg["job_id"],
                {"chunks": {}, "event": threading.Event(), "printed": False},
            )

            bucket["chunks"][msg["start_row"]] = np.array(msg["agg_shard"], dtype=float)
            bucket["event"].set()

            if len(bucket["chunks"]) < self.num_nodes or bucket["printed"]:
                return

            job = self.jobs.get(msg["job_id"])
            if job is None:
                return

            rows = []
            for start_row, end_row in job["row_parts"]:
                shard = bucket["chunks"].get(start_row)
                if shard is None:
                    shard = np.zeros((end_row - start_row, job["size"]), dtype=float)
                rows.append(shard)

            final_matrix = np.vstack(rows)
            bucket["printed"] = True

        print(f"[OptiReduce] Node {self.node_id} final matrix:")
        print(final_matrix)

    def optireduce_matrix(self, size, straggler_id=None, delay=0.0):
        if self.num_nodes is None:
            print("Not registered yet!")
            return

        if self.tb is None:
            self.calibrate_timeout()

        self.ensure_matrix(size)
        job_id = f"opti-{self.node_id}-{int(time.time() * 1000)}"
        row_parts = self.split_rows(size)

        self.results[job_id] = {
            "chunks": {},
            "event": threading.Event(),
            "printed": False,
        }

        start_msg = {
            "type": "OPTI_START",
            "job_id": job_id,
            "initiator_id": self.node_id,
            "size": size,
            "timeout": self.tb,
            "row_ranges": row_parts,
            "straggler_id": straggler_id,
            "delay": delay,
        }

        for peer_id, (ip, port) in enumerate(self.peers):
            if peer_id == self.node_id:
                continue
            self.send_msg(ip, port, start_msg, "tcp")

        self.start_round(start_msg)

    def run(self):
        self.start_server()

        print("Commands:")
        print("register")
        print("make_matrix <size>")
        print("show_matrix")
        print("calibrate")
        print("optireduce_matrix <size> [straggler_id] [delay_seconds]")
        print("send <node_id>")
        print("quit")

        while True:
            cmd = input("> ").strip()

            if cmd == "register":
                self.register()

            elif cmd.startswith("make_matrix"):
                parts = cmd.split()
                if len(parts) != 2:
                    print("Usage: make_matrix <size>")
                    continue
                self.make_matrix(int(parts[1]))

            elif cmd == "show_matrix":
                self.show_matrix()

            elif cmd == "calibrate":
                self.calibrate_timeout()

            elif cmd.startswith("optireduce_matrix"):
                parts = cmd.split()
                if len(parts) not in {2, 4}:
                    print("Usage: optireduce_matrix <size> [straggler_id] [delay_seconds]")
                    continue

                size = int(parts[1])
                straggler_id = int(parts[2]) if len(parts) == 4 else None
                delay = float(parts[3]) if len(parts) == 4 else 0.0
                self.optireduce_matrix(size, straggler_id, delay)

            elif cmd.startswith("send"):
                if self.num_nodes is None:
                    print("Not registered yet!")
                    continue

                parts = cmd.split()
                if len(parts) != 2:
                    print("Usage: send <node_id>")
                    continue

                self.send_random(int(parts[1]))

            elif cmd == "quit":
                print("Shutting down node...")
                sys.exit(0)

            else:
                print("Unknown command")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tcp", "udp"], required=True)
    args = parser.parse_args()

    node = OptiReduceNode(args.mode)
    node.run()
