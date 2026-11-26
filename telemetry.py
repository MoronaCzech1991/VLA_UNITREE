"""Viewer overlay and CSV telemetry for the simulated task."""
import csv
from pathlib import Path
import time

import mujoco as mj


class Telemetry:
    def __init__(self, viewer, model, data, addresses, mode, device, stream):
        self.viewer = viewer
        self.data = data
        self.addresses = addresses
        self.mode = mode
        self.device = device
        self.stage = "Preparing"
        self.instruction = ""
        self.episode = 0
        self.saved = 0
        self.last = -1.0
        self.start = time.monotonic()
        self.csv = csv.writer(stream)
        self.csv.writerow(["elapsed_s", "episode", "stage", "instruction", "saved",
                           "blue_x", "blue_y", "blue_z", "brown_x", "brown_y", "brown_z"])
        self.stream = stream

    def is_running(self):
        return self.viewer.is_running()

    def sync(self):
        now = time.monotonic() - self.start
        if self.last < 0 or now - self.last >= 0.12:
            self.last = now
            blue, brown = [self.data.qpos[a:a + 3] for a in self.addresses.values()]
            self.csv.writerow([round(now, 3), self.episode, self.stage, self.instruction,
                               self.saved, *blue, *brown])
            self.stream.flush()
            labels = "Mode\nPhysics\nEpisode\nStage\nSaved demos\nInstruction\nBlue XYZ (m)\nBrown XYZ (m)"
            values = "\n".join([self.mode, self.device, str(self.episode), self.stage,
                                 str(self.saved), self.instruction,
                                 "  ".join(f"{v:.2f}" for v in blue),
                                 "  ".join(f"{v:.2f}" for v in brown)])
            self.viewer.set_texts((mj.mjtFontScale.mjFONTSCALE_100,
                                   mj.mjtGridPos.mjGRID_BOTTOMLEFT, labels, values))
        self.viewer.sync()


def log_path():
    return Path("runs") / f"telemetry-{time.time_ns()}.csv"
