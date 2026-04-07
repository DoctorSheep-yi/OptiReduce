import time
import random
import subprocess
import threading


class Noise:
    def __init__(self):
        self.enable_straggler = False
        self.enable_packet_loss = False
        self.enable_bandwidth_limit = False
        self.enable_cpu_stress = False

        self.sleep_time = 0.0
        self.packet_loss_rate = 0.0
        self.bandwidth = None
        self.cpu_cores = 0

    # -------------------------
    # STRAGGLER (slow node)
    # -------------------------
    def apply_straggler(self):
        if self.enable_straggler:
            print(f"[NOISE] Straggler: sleep {self.sleep_time}s")
            time.sleep(self.sleep_time)

    # -------------------------
    # PACKET LOSS (UDP/TCP)
    # -------------------------
    # -------------------------
    # UDP LOSS (APP LEVEL)
    # -------------------------
    def should_drop_udp(self):
        if not self.enable_udp_loss:
            return False

        if random.random() < self.udp_loss_rate:
            print("[NOISE][UDP] Packet dropped")
            return True

        return False

    # -------------------------
    # TCP LOSS (SYSTEM LEVEL)
    # -------------------------
    def enable_tcp_packet_loss(self, loss="5%"):
        print(f"[NOISE][TCP] Applying packet loss {loss}")

        self.enable_tcp_loss = True
        self.tcp_loss_rate = loss

        # clear existing qdisc first
        subprocess.call(["tc", "qdisc", "del", "dev", "eth0", "root"],
                        stderr=subprocess.DEVNULL)

        cmd = [
            "tc", "qdisc", "add", "dev", "eth0",
            "root", "netem",
            "loss", loss
        ]
        subprocess.call(cmd)

    def clear_tcp_loss(self):
        print("[NOISE][TCP] Clearing packet loss")
        subprocess.call(["tc", "qdisc", "del", "dev", "eth0", "root"],
                        stderr=subprocess.DEVNULL)
        self.enable_tcp_loss = False

    # -------------------------
    # CPU STRESS
    # -------------------------
    def start_cpu_stress(self):
        if not self.enable_cpu_stress or self.cpu_cores <= 0:
            return

        print(f"[NOISE] CPU stress on {self.cpu_cores} cores")

        def stress():
            while True:
                pass

        for _ in range(self.cpu_cores):
            t = threading.Thread(target=stress, daemon=True)
            t.start()

    # -------------------------
    # BANDWIDTH LIMIT (tc)
    # -------------------------
    def apply_bandwidth_limit(self, rate="10mbit"):
        print(f"[NOISE] Limiting bandwidth to {rate}")
        cmd = [
            "tc", "qdisc", "add", "dev", "eth0",
            "root", "tbf",
            "rate", rate,
            "burst", "32kbit",
            "latency", "400ms"
        ]
        subprocess.call(cmd)

    def apply_packet_loss_tc(self, loss="5%"):
        print(f"[NOISE] Applying TCP packet loss {loss}")
        cmd = [
            "tc", "qdisc", "add", "dev", "eth0",
            "root", "netem",
            "loss", loss
        ]
        subprocess.call(cmd)

    def clear_tc(self):
        print("[NOISE] Clearing tc rules")
        subprocess.call(["tc", "qdisc", "del", "dev", "eth0", "root"])