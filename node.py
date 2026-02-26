import socket
import argparse
import random
import select
import struct
import time

PORT = 9000
TIMEOUT = 5000000  # 50 ms

def run_udp_send(peer):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    value = random.randint(1, 1000)
    msg = struct.pack("i", value)
    sock.sendto(msg, (peer, PORT))

    print(f"[UDP-SEND] sent value {value} to {peer}")

def run_udp_recv():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", PORT))

    r, _, _ = select.select([sock], [], [], TIMEOUT)
    if r:
        data, addr = sock.recvfrom(1024)
        value = struct.unpack("i", data)[0]
        print(f"[UDP-RECV] received value {value} from {addr}")
    else:
        print("[UDP-RECV] timeout, no value received")

def run_tcp_send(peer):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((peer, PORT))

    value = random.randint(1, 1000)
    msg = struct.pack("i", value)
    sock.sendall(msg)

    print(f"[TCP-SEND] sent value {value} to {peer}")
    sock.close()

def run_tcp_recv():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("", PORT))
    server.listen(1)

    conn, addr = server.accept()
    data = conn.recv(4)
    value = struct.unpack("i", data)[0]

    print(f"[TCP-RECV] received value {value} from {addr}")
    conn.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["udp-send", "udp-recv", "tcp-send", "tcp-recv"], required=True)
    ap.add_argument("--peer", help="receiver IP (send modes only)")
    args = ap.parse_args()

    if args.mode == "udp-send":
        run_udp_send(args.peer)
    elif args.mode == "udp-recv":
        run_udp_recv()
    elif args.mode == "tcp-send":
        run_tcp_send(args.peer)
    elif args.mode == "tcp-recv":
        run_tcp_recv()