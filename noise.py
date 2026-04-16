import time
import random
import subprocess

class Noise:
    def __init__(self):
        self.enable_straggler = False
        self.enable_udp_loss = False
        self.udp_loss_rate = 0.05
        self.sleep_time = 0.0

    def apply_straggler(self):
        if self.enable_straggler:
            print(f"[NOISE] Straggler: sleep {self.sleep_time}s")
            time.sleep(self.sleep_time)

    def should_drop_udp(self):
        if not self.enable_udp_loss: return False
        return random.random() < self.udp_loss_rate

    def apply_packet_loss_tc(self, loss="5%"):
        print(f"[NOISE] Applying system-level loss: {loss}")
        subprocess.call(["sudo", "tc", "qdisc", "del", "dev", "eth0", "root"], stderr=subprocess.DEVNULL)
        subprocess.call(["sudo", "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "loss", loss])

    def clear_tc(self):
        print("[NOISE] Clearing system-level loss")
        subprocess.call(["sudo", "tc", "qdisc", "del", "dev", "eth0", "root"], stderr=subprocess.DEVNULL)