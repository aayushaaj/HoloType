# -*- coding: utf-8 -*-
"""Shared runtime helpers for tracking, smoothing, logging, and metrics."""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .calibration import FINGER_MAP
from .decoder import NoisyChannelDecoder, RawSequenceDecoder
from .smoothing import FingertipKalmanFilter, MultiAxisOneEuroFilter
from .tap_detector import VelocityTapDetector
from .visuals import FINGER_COLORS


GENERIC_FINGERS = tuple(FINGER_COLORS.keys())
HAND_PREFIXES = ("left", "right")


@dataclass
class TextMetrics:
    expected: str
    actual: str
    char_error_rate: float
    word_error_rate: float
    accuracy: float
    wpm: float


class SessionLogger:
    """Append-only JSONL logger for replayable tap and output sessions."""

    def __init__(self, log_dir: str, mode: str, enabled: bool = True):
        self.enabled = enabled
        self.path: Optional[Path] = None
        if not enabled:
            return
        session_dir = Path(log_dir) / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = session_dir / f"{stamp}-{mode}.jsonl"

    def write(self, event_type: str, payload: Dict[str, Any]):
        if not self.enabled or self.path is None:
            return
        record = {"type": event_type, "timestamp": time.time(), **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def normalize_handedness(label: str) -> Optional[str]:
    lowered = (label or "").strip().lower()
    if lowered.startswith("left"):
        return "left"
    if lowered.startswith("right"):
        return "right"
    return None


def canonical_finger(handedness: str, finger: str, mode: str) -> str:
    if mode == "single_finger":
        return finger
    hand = normalize_handedness(handedness)
    return f"{hand}_{finger}" if hand else finger


def visual_finger_name(finger: str) -> str:
    for prefix in HAND_PREFIXES:
        marker = f"{prefix}_"
        if finger.startswith(marker):
            return finger[len(marker):]
    return finger


def display_finger_name(finger: Optional[str]) -> str:
    if not finger:
        return "-"
    return finger.replace("_", " ")


def configured_max_hands(mode: str, requested: int) -> int:
    if mode == "touch_typing":
        return max(requested, 2)
    return requested


def calibration_uses_handed_fingers(centroids: Dict[str, Dict[str, Any]]) -> bool:
    for centroid in centroids.values():
        finger = centroid.get("finger", "")
        if finger.startswith("left_") or finger.startswith("right_"):
            return True
    return False


def effective_mode_for_calibration(cal_mode: str, centroids: Dict[str, Dict[str, Any]]) -> str:
    if cal_mode == "touch_typing" and not calibration_uses_handed_fingers(centroids):
        return "single_finger"
    return cal_mode


def active_fingers_for_mode(mode: str) -> set[str]:
    if mode == "single_finger":
        return {"index"}
    return set(FINGER_MAP.keys())


def detection_finger_for_profile(handedness: str, raw_finger: str, mode: str,
                                 centroids: Dict[str, Dict[str, Any]]) -> str:
    if calibration_uses_handed_fingers(centroids):
        return canonical_finger(handedness, raw_finger, mode)
    return raw_finger


def create_smoother(config):
    if not config.smoothing.enabled:
        return None
    if config.smoothing.method == "kalman":
        return FingertipKalmanFilter(
            process_noise=config.smoothing.kalman_process_noise,
            measurement_noise=config.smoothing.kalman_measurement_noise,
        )
    return MultiAxisOneEuroFilter(
        min_cutoff=config.smoothing.one_euro_min_cutoff,
        beta=config.smoothing.one_euro_beta,
    )


def create_tap_detector(config, debug: bool = False) -> VelocityTapDetector:
    return VelocityTapDetector(
        history_len=config.tap_detection.history_len,
        down_vel_thresh=config.tap_detection.down_vel_thresh,
        up_vel_thresh=config.tap_detection.up_vel_thresh,
        pip_margin=config.tap_detection.pip_margin,
        refractory=config.tap_detection.refractory,
        min_press_duration=config.tap_detection.min_press_duration,
        max_press_duration=config.tap_detection.max_press_duration,
        debug=debug,
    )


def create_decoder(config):
    vocabulary_path = config.vocabulary_path or config.decoder.vocabulary_path or None
    if config.decoder.method == "spellchecker":
        return RawSequenceDecoder()
    return NoisyChannelDecoder(
        vocabulary_path=vocabulary_path,
        ngram_order=config.decoder.ngram_order,
        emission_weight=config.decoder.emission_weight,
        lm_weight=config.decoder.lm_weight,
        freq_weight=config.decoder.freq_weight,
    )


def feed_decoder(decoder, key: str, finger: str, xy_dist: float, z_dist: float,
                 confidence: float, timestamp: float):
    if isinstance(decoder, RawSequenceDecoder):
        decoder.feed_key(key, confidence=confidence)
    else:
        decoder.feed_key(
            key=key,
            finger=finger,
            xy_dist=xy_dist,
            z_dist=z_dist,
            tap_confidence=confidence,
            timestamp=timestamp,
        )


def calibration_is_usable(metrics: Dict[str, Any], mode: str, min_samples_per_key: int) -> tuple[bool, str]:
    completion = metrics.get("completion", 0.0)
    if completion < 1.0:
        return False, f"calibration incomplete ({completion:.0%})"

    key_counts = metrics.get("key_counts", {})
    weak_keys = [key for key, count in key_counts.items() if count < min_samples_per_key]
    if weak_keys:
        shown = ", ".join(repr(key) for key in weak_keys[:8])
        return False, f"not enough samples for keys {shown}"

    finger_metrics = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, dict) and k != "key_counts"
    }
    if not finger_metrics:
        return False, "no calibration samples recorded"

    required = ["index"] if mode == "single_finger" else [
        finger for finger, values in finger_metrics.items() if values.get("unique_keys", 0) > 0
    ]
    weak = [
        finger for finger in required
        if finger_metrics.get(finger, {}).get("n_samples", 0) < min_samples_per_key
    ]
    if weak:
        names = ", ".join(display_finger_name(finger) for finger in weak[:4])
        return False, f"not enough samples for {names}"
    return True, "ok"


def edit_distance(left: Iterable[Any], right: Iterable[Any]) -> int:
    a = list(left)
    b = list(right)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a):
        current = [i + 1]
        for j, item_b in enumerate(b):
            current.append(
                min(
                    previous[j + 1] + 1,
                    current[j] + 1,
                    previous[j] + (item_a != item_b),
                )
            )
        previous = current
    return previous[-1]


def score_text(expected: str, actual: str, duration_seconds: float = 0.0) -> TextMetrics:
    expected = " ".join((expected or "").split())
    actual = " ".join((actual or "").split())
    char_den = max(len(expected), 1)
    expected_words = expected.split()
    actual_words = actual.split()
    word_den = max(len(expected_words), 1)
    char_errors = edit_distance(expected, actual)
    word_errors = edit_distance(expected_words, actual_words)
    minutes = max(duration_seconds / 60.0, 1e-9)
    wpm = len(actual_words) / minutes if duration_seconds > 0 else 0.0
    return TextMetrics(
        expected=expected,
        actual=actual,
        char_error_rate=char_errors / char_den,
        word_error_rate=word_errors / word_den,
        accuracy=max(0.0, 1.0 - char_errors / char_den),
        wpm=wpm,
    )


def load_jsonl(path: str) -> list[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def metrics_to_dict(metrics: TextMetrics) -> Dict[str, Any]:
    return asdict(metrics)
