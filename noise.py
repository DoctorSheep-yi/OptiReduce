import time
import random
import subprocess
import threading

class Noise:
    def __init__(self):
        self.enable_straggler = False
        self.enable_udp_loss = False     # Added missing attribute
        self.enable_tcp_loss = False
        self.enable_cpu_stress = False

        self.sleep_time = 0.0
        self.udp_loss_rate = 0.0        # Added missing attribute
        self.tcp_loss_rate = "0%"
        self.cpu_cores = 0

    def apply_straggler(self):
        if self.enable_straggler:
            print(f"[NOISE] Straggler: sleep {self.sleep_time}s")
            time.sleep(self.sleep_time)

    def should_drop_udp(self):
        if not self.enable_udp_loss:
            return False
        if random.random() < self.udp_loss_rate:
            print("[NOISE][UDP] Packet dropped")
            return True
        return False

    def apply_packet_loss_tc(self, loss="5%"):
        print(f"[NOISE] Applying system-level loss: {loss}")
        subprocess.call(["sudo", "tc", "qdisc", "del", "dev", "eth0", "root"], stderr=subprocess.DEVNULL)
        subprocess.call(["sudo", "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "loss", loss])
        self.enable_tcp_loss = True
        self.tcp_loss_rate = loss

    def clear_tc(self):
        print("[NOISE] Clearing system-level loss")
        subprocess.call(["sudo", "tc", "qdisc", "del", "dev", "eth0", "root"], stderr=subprocess.DEVNULL)
        self.enable_tcp_loss = False

    def start_cpu_stress(self, cores=1):
        self.cpu_cores = cores
        def stress():
            while True: pass
        for _ in range(self.cpu_cores):
            threading.Thread(target=stress, daemon=True).start()