import socket
import threading
import pickle

PORT = 8000

nodes = []   # [(ip, port)]
lock = threading.Lock()


def broadcast():
    dead_nodes = []

    for idx, (ip, port) in enumerate(nodes):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))

            msg = {
                "type": "PEER_UPDATE",
                "node_id": idx,
                "peers": nodes
            }

            sock.sendall(pickle.dumps(msg))
            sock.close()

        except Exception as e:
            print(f"[Coordinator] Failed to send to {ip}:{port}: {e}")
            dead_nodes.append((ip, port))

    # remove dead nodes automatically
    if dead_nodes:
        with lock:
            for n in dead_nodes:
                if n in nodes:
                    nodes.remove(n)

        print(f"[Coordinator] Cleaned dead nodes: {dead_nodes}")

        # rebroadcast updated list
        if nodes:
            broadcast()


def handle_conn(conn, addr):
    data = conn.recv(4096)

    try:
        msg = pickle.loads(data)

        if msg["type"] == "REGISTER":
            ip = addr[0]
            port = msg["port"]

            with lock:
                if (ip, port) not in nodes:
                    nodes.append((ip, port))
                    nodes.sort()

            print(f"[Coordinator] Registered: {ip}:{port}")
            print(f"[Coordinator] Nodes: {nodes}")

            broadcast()

        elif msg["type"] == "UNREGISTER":
            ip = addr[0]
            port = msg["port"]

            with lock:
                if (ip, port) in nodes:
                    nodes.remove((ip, port))

            print(f"[Coordinator] Unregistered: {ip}:{port}")
            print(f"[Coordinator] Nodes: {nodes}")

            broadcast()

    except Exception as e:
        print("Error:", e)

    conn.close()


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", PORT))
    server.listen()

    print(f"[Coordinator] Running on port {PORT}")

    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_conn, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()