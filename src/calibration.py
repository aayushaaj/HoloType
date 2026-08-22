# -*- coding: utf-8 -*-
"""Calibration - Multi-finger touch-typing calibration with centroid computation."""

import json
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

FINGER_MAP = {
    "left_pinky":  list("qaz") + ["`", "1", "tab", "caps", "shift"],
    "left_ring":   list("wsx") + ["2"],
    "left_middle": list("edc") + ["3"],
    "left_index":  list("rfvtgb") + ["4", "5"],
    "right_index": list("yhnujm") + ["6", "7"],
    "right_middle": list("ik,") + ["8"],
    "right_ring":  list("ol.") + ["9"],
    "right_pinky": list("p;/'") + ["0", "-", "=", "[", "]", "\\", "enter", "shift"],
    "left_thumb":  [" "],
    "right_thumb": [" "],
}

CALIBRATION_PHRASES = {
    "single_finger": "the quick brown fox jumps over lazy dog",
    "touch_typing": "the quick brown fox jumps over the lazy dog. packing boxes with zest. very quick zombies jump.",
    "full_coverage": "the quick brown fox jumps over the lazy dog. packing my box with five dozen liquor jugs. how vexingly quick daft zebras jump!",
}

CALIBRATION_PHRASE = CALIBRATION_PHRASES["single_finger"]

FINGER_COLUMNS = {
    "pinky": list("qaz") + ["p", ";", "/"],
    "ring": list("wsx") + ["o", "l"],
    "middle": list("edc") + ["i", "k", ","],
    "index": list("rfvtgb") + list("yhnujm"),
    "thumb": [" "],
}


@dataclass
class KeySample:
    key: str
    finger: str
    hand: str
    x: float
    y: float
    z: float
    timestamp: float
    confidence: float = 1.0


@dataclass
class FingerCalibration:
    finger: str
    hand: str
    samples: List[KeySample]
    centroid: Optional[Dict] = None

    def compute_centroid(self) -> Dict:
        if not self.samples:
            return None
        by_key = defaultdict(list)
        for s in self.samples:
            by_key[s.key].append(s)

        centroids = {}
        for key, samples in by_key.items():
            xs = [s.x for s in samples]
            ys = [s.y for s in samples]
            zs = [s.z for s in samples]
            confs = [s.confidence for s in samples]
            total_conf = sum(confs)
            centroids[key] = {
                "x": sum(x * c for x, c in zip(xs, confs)) / total_conf,
                "y": sum(y * c for y, c in zip(ys, confs)) / total_conf,
                "z": sum(z * c for z, c in zip(zs, confs)) / total_conf,
                "n_samples": len(samples),
                "confidence": total_conf / len(samples),
            }
        self.centroid = centroids
        return centroids


class MultiFingerCalibrator:
    def __init__(self, mode: str = "touch_typing"):
        self.mode = mode
        self.phrase = CALIBRATION_PHRASES.get(mode, CALIBRATION_PHRASES["touch_typing"])
        self.finger_calibrations: Dict[str, FingerCalibration] = {}
        self._init_fingers()
        self.current_key_idx = 0
        self.current_finger = None

    def _init_fingers(self):
        for finger in FINGER_MAP.keys():
            hand = "left" if finger.startswith("left_") else "right"
            self.finger_calibrations[finger] = FingerCalibration(finger=finger, hand=hand, samples=[])

    def get_next_target(self) -> Optional[Tuple[str, str]]:
        if self.current_key_idx >= len(self.phrase):
            return None
        key = self.phrase[self.current_key_idx]
        if key == " ":
            finger = "left_thumb" if self.current_key_idx % 2 == 0 else "right_thumb"
        else:
            finger = self._find_finger_for_key(key)
        self.current_finger = finger
        return key, finger

    def _find_finger_for_key(self, key: str) -> str:
        key_lower = key.lower()
        for finger, keys in FINGER_MAP.items():
            if key_lower in keys:
                return finger
        return "left_index"

    def record_sample(self, key: str, finger: str, x: float, y: float, z: float, confidence: float = 1.0):
        if finger not in self.finger_calibrations:
            return
        hand = "left" if finger.startswith("left_") else "right"
        sample = KeySample(key, finger, hand, x, y, z, time.time(), confidence)
        self.finger_calibrations[finger].samples.append(sample)
        self.current_key_idx += 1

    def build_all_centroids(self) -> Dict:
        all_centroids = {}
        for finger, calib in self.finger_calibrations.items():
            centroids = calib.compute_centroid()
            if centroids:
                for key, data in centroids.items():
                    data["finger"] = finger
                    data["hand"] = calib.hand
                    all_centroids[key] = data
        return all_centroids

    def get_quality_metrics(self) -> Dict:
        metrics = {}
        total_samples = 0
        for finger, calib in self.finger_calibrations.items():
            n = len(calib.samples)
            unique_keys = len(set(s.key for s in calib.samples))
            avg_conf = np.mean([s.confidence for s in calib.samples]) if n > 0 else 0
            metrics[finger] = {"n_samples": n, "unique_keys": unique_keys, "avg_confidence": avg_conf}
            total_samples += n
        metrics["total_samples"] = total_samples
        metrics["completion"] = self.current_key_idx / len(self.phrase) if self.phrase else 1.0
        return metrics

    def save(self, path: str):
        centroids = self.build_all_centroids()
        data = {
            "mode": self.mode,
            "phrase": self.phrase,
            "centroids": centroids,
            "finger_metrics": {f: {"n_samples": len(c.samples)} for f, c in self.finger_calibrations.items()},
            "timestamp": time.time(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> Dict:
        with open(path) as f:
            return json.load(f)


def nearest_key(x: float, y: float, z: float, finger: str, centroids: Dict, use_hand: bool = True) -> Tuple[str, float, float]:
    plausible = []
    finger_hand = "left" if finger.startswith("left_") else "right"

    for key, c in centroids.items():
        c_finger = c.get("finger", "")
        c_hand = c.get("hand", "")
        if finger != c_finger:
            continue
        if use_hand and c_hand != finger_hand and key != " ":
            continue
        plausible.append((key, c))

    if not plausible:
        plausible = [(k, c) for k, c in centroids.items()]

    best_key, best_xy_dist, best_z_dist = None, float("inf"), float("inf")
    for key, c in plausible:
        xy_dist = ((x - c["x"])**2 + (y - c["y"])**2)**0.5
        z_dist = abs(z - c.get("z", 0))
        if xy_dist < best_xy_dist:
            best_key, best_xy_dist, best_z_dist = key, xy_dist, z_dist

    return best_key, best_xy_dist, best_z_dist


if __name__ == "__main__":
    print("Available calibration modes:")
    for mode, phrase in CALIBRATION_PHRASES.items():
        print(f"  {mode}: {phrase[:60]}... ({len(phrase)} chars)")
    print(f"\nFinger map keys: {list(FINGER_MAP.keys())}")