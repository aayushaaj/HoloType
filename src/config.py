# -*- coding: utf-8 -*-
"""Config - Centralized configuration with validation and persistence."""

from dataclasses import dataclass, field, fields
from typing import Any
import json
import os


@dataclass
class HandTrackingConfig:
    max_hands: int = 1
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.7
    model_complexity: int = 1


@dataclass
class SmoothingConfig:
    enabled: bool = True
    method: str = "kalman"
    kalman_process_noise: float = 1e-4
    kalman_measurement_noise: float = 1e-3
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.0


@dataclass
class TapDetectionConfig:
    method: str = "velocity"
    history_len: int = 8
    down_vel_thresh: float = -0.012
    up_vel_thresh: float = 0.008
    pip_margin: float = 0.015
    refractory: float = 0.12
    min_press_duration: float = 0.02
    max_press_duration: float = 0.3


@dataclass
class CalibrationConfig:
    mode: str = "touch_typing"
    phrase: str = ""
    min_samples_per_key: int = 1
    use_regression: bool = False


@dataclass
class DecoderConfig:
    method: str = "noisy_channel"
    vocabulary_path: str = ""
    ngram_order: int = 3
    emission_weight: float = 1.0
    lm_weight: float = 0.5
    freq_weight: float = 0.3


@dataclass
class VisualConfig:
    show_keyboard: bool = True
    show_skeleton: bool = False
    show_finger_trails: bool = True
    show_confidence_ripples: bool = True
    keyboard_opacity: float = 0.88
    target_key_pulse: bool = True


@dataclass
class AppConfig:
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    tap_detection: TapDetectionConfig = field(default_factory=TapDetectionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)

    calibration_path: str = "data/calibration_profile.json"
    vocabulary_path: str = "data/vocabulary.txt"
    log_dir: str = "logs"

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    def save(self, path: str):
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    def _to_dict(self) -> dict:
        def dc_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dc_to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        return dc_to_dict(self)

    @classmethod
    def _from_dict(cls, data: dict) -> "AppConfig":
        def dict_to_dc(d, dc_type):
            if not isinstance(d, dict):
                return d
            field_types = {f.name: f.type for f in fields(dc_type)}
            kwargs = {}
            for k, v in d.items():
                if k in field_types:
                    field_type = field_types[k]
                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[k] = dict_to_dc(v, field_type)
                    else:
                        kwargs[k] = v
            return dc_type(**kwargs)

        return cls(
            hand_tracking=dict_to_dc(data.get("hand_tracking", {}), HandTrackingConfig),
            smoothing=dict_to_dc(data.get("smoothing", {}), SmoothingConfig),
            tap_detection=dict_to_dc(data.get("tap_detection", {}), TapDetectionConfig),
            calibration=dict_to_dc(data.get("calibration", {}), CalibrationConfig),
            decoder=dict_to_dc(data.get("decoder", {}), DecoderConfig),
            visual=dict_to_dc(data.get("visual", {}), VisualConfig),
            calibration_path=data.get("calibration_path", "data/calibration_profile.json"),
            vocabulary_path=data.get("vocabulary_path", "data/vocabulary.txt"),
            log_dir=data.get("log_dir", "logs"),
        )


DEFAULT_CONFIG = AppConfig()