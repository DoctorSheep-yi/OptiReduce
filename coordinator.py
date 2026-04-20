# coordinator.py (FIXED VERSION)

import socket
import threading
import pickle

PORT = 8000

nodes = []          # [(ip, port)]
node_ids = {}       # (ip, port) -> id
next_id = 0

lock = threading.Lock()


def broadcast():
    for (ip, port) in nodes:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))

            msg = {
                "type": "PEER_UPDATE",
                "node_id": node_ids[(ip, port)],
                "peers": nodes
            }

            data = pickle.dumps(msg)
            sock.sendall(len(data).to_bytes(4, 'big') + data)
            sock.close()
        except Exception as e:
            print(f"[Coordinator] Failed to send to {ip}:{port}: {e}")


def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

def handle_conn(conn, addr):
    global next_id

    try:
        # 2. First, read the 4-byte length prefix
        length_bytes = recv_exact(conn, 4)
        if not length_bytes:
            conn.close()
            return

        length = int.from_bytes(length_bytes, 'big')
        
        # 3. Then read exactly 'length' bytes
        data = recv_exact(conn, length)
        if not data:
            conn.close()
            return

        msg = pickle.loads(data)

        if msg["type"] == "REGISTER":
            ip = addr[0]
            port = msg["port"]
            node = (ip, port)

            with lock:
                if node not in nodes:
                    nodes.append(node)
                    node_ids[node] = next_id
                    print(f"[Coordinator] Assign ID {next_id} to {node}")
                    next_id += 1

            print(f"[Coordinator] Nodes: {nodes}")
            broadcast()
            
        # ... rest of your code ...

    except Exception as e:
        print("Error:", e)
    finally:
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