import time
import random
import subprocess
from multiprocessing import Process


class Noise:
    def __init__(self):
        self.enable_straggler = False
        self.enable_udp_loss = False

        self.udp_loss_rate = 0.05
        self.sleep_time = 0.0

        # CPU stress
        self.cpu_stress = False
        self.cpu_workers = 1
        self._stress_processes = []
        self._stop_flag = False

    # =========================
    # STRAGGLER DELAY
    # =========================
    def apply_straggler(self):
        if self.enable_straggler:
            time.sleep(self.sleep_time)

    # =========================
    # CPU STRESS (REAL)
    # =========================
    def _burn_cpu(self):
        while True:
            x = 0
            for _ in range(10_000_000):
                x += 1

    def start_cpu_stress(self):
        if not self.cpu_stress:
            return

        self._stress_processes = []

        for _ in range(self.cpu_workers):
            p = Process(target=self._burn_cpu)
            p.daemon = True
            p.start()
            self._stress_processes.append(p)

    def stop_cpu_stress(self):
        for p in self._stress_processes:
            p.terminate()
        self._stress_processes = []

    # =========================
    # UDP LOSS
    # =========================
    def should_drop_udp(self):
        if not self.enable_udp_loss:
            return False
        return random.random() < self.udp_loss_rate

    # =========================
    # SYSTEM LOSS (tc)
    # =========================
    def apply_packet_loss_tc(self, loss="5%"):
        subprocess.call(["sudo", "tc", "qdisc", "del", "dev", "eth0", "root"], stderr=subprocess.DEVNULL)
        subprocess.call(["sudo", "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "loss", loss])

    def clear_tc(self):
        subprocess.call(["sudo", "tc", "qdisc", "del", "dev", "eth0", "root"], stderr=subprocess.DEVNULL)