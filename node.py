import socket
import argparse
import random
import struct
import threading

PORT = 9000
stop_event = threading.Event()


# ================= UDP =================

def udp_recv():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", PORT))

    while not stop_event.is_set():
        data, addr = sock.recvfrom(1024)
        value = struct.unpack("i", data)[0]
        print(f"[UDP-RECV] {value} from {addr}")


def udp_send(ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    value = random.randint(1, 1000)
    msg = struct.pack("i", value)

    sock.sendto(msg, (ip, PORT))
    print(f"[UDP-SEND] {value} -> {ip}")


# ================= TCP =================

def tcp_recv():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("", PORT))
    server.listen()

    while not stop_event.is_set():
        conn, addr = server.accept()

        data = conn.recv(4)
        if data:
            value = struct.unpack("i", data)[0]
            print(f"[TCP-RECV] {value} from {addr}")

        conn.close()


def tcp_send(ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, PORT))

    value = random.randint(1, 1000)
    msg = struct.pack("i", value)

    sock.sendall(msg)
    print(f"[TCP-SEND] {value} -> {ip}")

    sock.close()


# ================= MAIN =================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["udp", "tcp"], required=True)
    args = ap.parse_args()

    # start receiver thread
    if args.mode == "udp":
        threading.Thread(target=udp_recv, daemon=True).start()
    else:
        threading.Thread(target=tcp_recv, daemon=True).start()

    print("Commands:")
    print("send <ip>")
    print("quit")

    while True:
        cmd = input("> ").strip()

        if cmd == "quit":
            stop_event.set()
            break

        if cmd.startswith("send"):
            parts = cmd.split()

            if len(parts) != 2:
                print("Usage: send <ip>")
                continue

            ip = parts[1]

            if args.mode == "udp":
                udp_send(ip)
            else:
                tcp_send(ip)

    print("Program exited.")