
"""TapDetector - Z-velocity based tap detection with very sensitive defaults."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TapEvent:
    finger: str
    x: float
    y: float
    z: float
    timestamp: float
    confidence: float
    features: Dict


class VelocityTapDetector:
    def __init__(
        self,
        history_len: int = 8,
        down_vel_thresh: float = -0.004,
        up_vel_thresh: float = 0.003,
        pip_margin: float = 0.005,
        refractory: float = 0.08,
        min_press_duration: float = 0.01,
        max_press_duration: float = 0.6,
        debug: bool = False,
    ):
        self.history_len = history_len
        self.down_thresh = down_vel_thresh
        self.up_thresh = up_vel_thresh
        self.pip_margin = pip_margin
        self.refractory = refractory
        self.min_dur = min_press_duration
        self.max_dur = max_press_duration
        self.debug = debug

        fingers = ["left_pinky", "left_ring", "left_middle", "left_index", "left_thumb",
                   "right_pinky", "right_ring", "right_middle", "right_index", "right_thumb",
                   "thumb", "index", "middle", "ring", "pinky"]

        self.pos_history = {f: deque(maxlen=history_len) for f in fingers}
        self.pip_history = {f: deque(maxlen=history_len) for f in fingers}
        self.vel_history = {f: deque(maxlen=history_len) for f in fingers}
        self.accel_history = {f: deque(maxlen=history_len) for f in fingers}

        self.armed = {f: False for f in fingers}
        self.arm_time = {f: 0.0 for f in fingers}
        self.last_tap = {f: 0.0 for f in fingers}

        self.adaptive_down_thresh = {f: down_vel_thresh for f in fingers}
        self.adaptive_up_thresh = {f: up_vel_thresh for f in fingers}

    def update(self, finger: str, tip_x: float, tip_y: float, tip_z: float,
               pip_x: float, pip_y: float, pip_z: float, timestamp: float) -> Optional[TapEvent]:
        hist = self.pos_history[finger]
        hist.append((tip_x, tip_y, tip_z, timestamp))

        pip_hist = self.pip_history[finger]
        pip_hist.append((pip_x, pip_y, pip_z, timestamp))

        if len(hist) < 3:
            return None

        if timestamp - self.last_tap[finger] < self.refractory:
            return None

        vel, accel = self._compute_derivatives(hist)
        self.vel_history[finger].append(vel)
        self.accel_history[finger].append(accel)

        vx, vy, vz = vel
        ax, ay, az = accel

        if pip_hist:
            _, _, pip_z_curr, _ = pip_hist[-1]
            z_rel = tip_z - pip_z_curr
        else:
            z_rel = 0

        if not self.armed[finger]:
            if vz < self.adaptive_down_thresh[finger] and z_rel < -self.pip_margin:
                self.armed[finger] = True
                self.arm_time[finger] = timestamp
                if self.debug:
                    print(f"[DEBUG] {finger} ARMED: vz={vz:.5f}, z_rel={z_rel:.5f}")
            elif self.debug and vz < -0.001:
                print(f"[DEBUG] {finger} near-threshold: vz={vz:.5f}, z_rel={z_rel:.5f}, need vz<{self.adaptive_down_thresh[finger]:.5f} z_rel<{-self.pip_margin:.5f}")
            return None
        else:
            press_duration = timestamp - self.arm_time[finger]

            if press_duration > self.max_dur:
                self.armed[finger] = False
                if self.debug:
                    print(f"[DEBUG] {finger} timeout ({press_duration:.3f}s)")
                return None

            if vz > self.adaptive_up_thresh[finger] and press_duration > self.min_dur:
                self.armed[finger] = False
                self.last_tap[finger] = timestamp

                confidence = self._compute_confidence(vz, z_rel, press_duration, az, finger)
                self._adapt_thresholds(finger, vz, confidence)

                if self.debug:
                    print(f"[DEBUG] {finger} TAP! vz={vz:.5f}, z_rel={z_rel:.5f}, dur={press_duration:.3f}, conf={confidence:.2f}")

                return TapEvent(
                    finger=finger, x=tip_x, y=tip_y, z=tip_z,
                    timestamp=timestamp, confidence=confidence,
                    features={"vz_down": vz, "z_rel": z_rel, "duration": press_duration, "az": az}
                )
            elif self.debug and press_duration > self.min_dur:
                print(f"[DEBUG] {finger} waiting rebound: vz={vz:.5f}, need >{self.adaptive_up_thresh[finger]:.5f}")
            return None

    def _compute_derivatives(self, hist: deque) -> tuple:
        if len(hist) < 2:
            return (0, 0, 0), (0, 0, 0)
        pts = list(hist)[-3:]
        (x1, y1, z1, t1), (x2, y2, z2, t2), (x3, y3, z3, t3) = pts
        dt1 = max(t2 - t1, 1e-6)
        dt2 = max(t3 - t2, 1e-6)
        vx1, vy1, vz1 = (x2-x1)/dt1, (y2-y1)/dt1, (z2-z1)/dt1
        vx2, vy2, vz2 = (x3-x2)/dt2, (y3-y2)/dt2, (z3-z2)/dt2
        vx, vy, vz = (vx1+vx2)/2, (vy1+vy2)/2, (vz1+vz2)/2
        dt_avg = (dt1 + dt2) / 2
        ax, ay, az = (vx2-vx1)/dt_avg, (vy2-vy1)/dt_avg, (vz2-vz1)/dt_avg
        return (vx, vy, vz), (ax, ay, az)

    def _compute_confidence(self, vz: float, z_rel: float, duration: float,
                           az: float, finger: str) -> float:
        vel_conf = min(abs(vz) / (abs(self.adaptive_down_thresh[finger]) * 2), 1.0)
        depth_conf = min(abs(z_rel) / (self.pip_margin * 3), 1.0)
        ideal_dur = 0.08
        dur_conf = 1.0 - min(abs(duration - ideal_dur) / ideal_dur, 1.0)
        accel_conf = min(abs(az) / 0.5, 1.0)
        confidence = 0.35 * vel_conf + 0.25 * depth_conf + 0.2 * dur_conf + 0.2 * accel_conf
        return max(0.1, min(1.0, confidence))

    def _adapt_thresholds(self, finger: str, vz: float, confidence: float):
        if confidence > 0.7:
            self.adaptive_down_thresh[finger] *= 0.995
            self.adaptive_up_thresh[finger] *= 0.995
        elif confidence < 0.3:
            self.adaptive_down_thresh[finger] *= 1.005
            self.adaptive_up_thresh[finger] *= 1.005
        self.adaptive_down_thresh[finger] = np.clip(self.adaptive_down_thresh[finger], -0.02, -0.002)
        self.adaptive_up_thresh[finger] = np.clip(self.adaptive_up_thresh[finger], 0.001, 0.01)


class LearnedTapDetector:
    def __init__(self, model_path: str = None):
        self.model = None
        self.window_size = 10
        self.feature_buffers = {}
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        pass

    def update(self, finger: str, features: Dict) -> Optional[TapEvent]:
        pass

    def collect_training_sample(self, finger: str, features: List[Dict], label: int):
        pass