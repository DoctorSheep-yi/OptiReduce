import time
import random
import subprocess
import threading

class Noise:
    def __init__(self):
        self.enable_straggler = False
        self.enable_udp_loss = False  # Fixed: Added missing attribute
        self.enable_tcp_loss = False
        self.enable_cpu_stress = False

        self.sleep_time = 0.0
        self.udp_loss_rate = 0.0     # Fixed: Added missing attribute
        self.tcp_loss_rate = "0%"
        self.cpu_cores = 0

    # -------------------------
    # STRAGGLER (Computational Variability)
    # -------------------------
    def apply_straggler(self):
        if self.enable_straggler and self.sleep_time > 0:
            print(f"[NOISE] Straggler: sleeping {self.sleep_time}s")
            time.sleep(self.sleep_time)

    # -------------------------
    # UDP LOSS (Application Level)
    # -------------------------
    def should_drop_udp(self):
        if not self.enable_udp_loss:
            return False
        return random.random() < self.udp_loss_rate

    # -------------------------
    # TC PACKET LOSS (System Level - Affects All Traffic)
    # -------------------------
    def apply_packet_loss_tc(self, loss="5%"):
        """Uses Linux Traffic Control to drop packets at the kernel level."""
        print(f"[NOISE] Applying kernel-level packet loss: {loss}")
        self.clear_tc()
        # netem is the network emulator for simulating packet loss
        cmd = ["sudo", "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "loss", loss]
        subprocess.call(cmd)
        self.enable_tcp_loss = True
        self.tcp_loss_rate = loss

    def clear_tc(self):
        """Removes all traffic control rules."""
        print("[NOISE] Clearing all network noise (tc)")
        subprocess.call(["sudo", "tc", "qdisc", "del", "dev", "eth0", "root"], 
                        stderr=subprocess.DEVNULL)
        self.enable_tcp_loss = False

    # -------------------------
    # CPU STRESS
    # -------------------------
    def start_cpu_stress(self, cores=1):
        self.cpu_cores = cores
        print(f"[NOISE] Starting CPU stress on {self.cpu_cores} cores")
        def stress():
            while True:
                pass # Simple infinite loop to consume CPU
        for _ in range(self.cpu_cores):
            threading.Thread(target=stress, daemon=True).start()