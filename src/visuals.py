# -*- coding: utf-8 -*-
"""GlassmorphismRenderer - Lightweight iOS-style glass keyboard (60 FPS optimized)."""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class KeyState(Enum):
    IDLE = "idle"
    HOVER = "hover"
    PRESSED = "pressed"
    TARGET = "target"


@dataclass
class KeyVisualConfig:
    key: str
    x: float
    y: float
    z: float = 0.0
    finger: str = "index"
    width: float = 0.08
    height: float = 0.09
    state: KeyState = KeyState.IDLE
    press_progress: float = 0.0
    confidence: float = 0.0


@dataclass
class FingerVisualState:
    name: str
    x: float
    y: float
    z: float
    color: Tuple[int, int, int]
    visible: bool = True
    trail: List[Tuple[float, float]] = field(default_factory=list)


FINGER_COLORS = {
    "pinky": (255, 80, 80),
    "ring": (255, 165, 40),
    "middle": (60, 205, 85),
    "index": (60, 160, 255),
    "thumb": (185, 100, 255),
}

FINGER_COLUMNS = {
    "pinky": list("qaz") + ["p", ";", "/"],
    "ring": list("wsx") + ["o", "l"],
    "middle": list("edc") + ["i", "k", ","],
    "index": list("rfvtgb") + list("yhnujm"),
    "thumb": [" "],
}


class GlassmorphismRenderer:
    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.key_radius = 14
        self._init_key_layout()
        self._frame_count = 0

    def _init_key_layout(self):
        self.key_configs: Dict[str, KeyVisualConfig] = {}
        rows = [
            (list("qwertyuiop"), 0.38),
            (list("asdfghjkl"), 0.46),
            (list("zxcvbnm"), 0.54),
        ]
        pad = 0.035
        span = 1.0 - 2 * pad

        for row_idx, (row, y) in enumerate(rows):
            n = len(row)
            for i, key in enumerate(row):
                x = pad + (i + 0.5) * (span / n)
                finger = self._get_finger_for_key(key)
                self.key_configs[key] = KeyVisualConfig(
                    key=key, x=x, y=y, finger=finger,
                    width=0.082 if row_idx < 2 else 0.078,
                    height=0.078
                )
        self.key_configs[" "] = KeyVisualConfig(
            key=" ", x=0.5, y=0.72, finger="thumb", width=0.28, height=0.078
        )

    def _get_finger_for_key(self, key: str) -> str:
        for finger, keys in FINGER_COLUMNS.items():
            if key in keys:
                return finger
        return "index"

    def update_key_state(self, key: str, state: KeyState, progress: float = 0.0, confidence: float = 0.0):
        if key in self.key_configs:
            self.key_configs[key].state = state
            self.key_configs[key].press_progress = np.clip(progress, 0, 1)
            self.key_configs[key].confidence = np.clip(confidence, 0, 1)

    def render(self, frame: np.ndarray,
               finger_states: Optional[Dict[str, FingerVisualState]] = None,
               target_key: Optional[str] = None,
               show_skeleton: bool = False,
               hand_landmarks: Optional[List] = None) -> np.ndarray:
        self._frame_count += 1
        h, w = frame.shape[:2]

        overlay = frame.copy()

        for key, config in self.key_configs.items():
            self._draw_key(overlay, config, w, h, target_key)

        if finger_states and target_key and target_key in self.key_configs:
            self._draw_finger_guides(overlay, finger_states, target_key, w, h)

        if finger_states:
            self._draw_fingertips(overlay, finger_states, w, h)

        if show_skeleton and hand_landmarks:
            self._draw_skeleton(overlay, hand_landmarks, w, h)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame

    def _draw_key(self, img: np.ndarray, config: KeyVisualConfig, w: int, h: int, target_key: str = None):
        cx = int(config.x * w)
        cy = int(config.y * h)
        kw = int(config.width * w)
        kh = int(config.height * h)
        x1 = cx - kw // 2
        y1 = cy - kh // 2
        x2 = cx + kw // 2
        y2 = cy + kh // 2
        r = self.key_radius

        if config.press_progress > 0:
            scale = 1.0 + 0.05 * config.press_progress
            nkw, nkh = int(kw * scale), int(kh * scale)
            x1, x2 = cx - nkw // 2, cx + nkw // 2
            y1, y2 = cy - nkh // 2, cy + nkh // 2

        finger_color = FINGER_COLORS.get(config.finger, (180, 180, 180))
        is_target = target_key is not None and config.key == target_key

        self._draw_glass_key(img, x1, y1, x2, y2, r, finger_color, config.state,
                              config.press_progress, config.confidence, is_target)

        label = config.key.upper() if config.key != " " else "SPACE"
        font_scale = 0.65 if config.key != " " else 0.52
        weight = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, weight)
        tx = cx - tw // 2
        ty = cy + th // 2 - 1

        cv2.putText(img, label, (tx + 1, ty + 1), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    (0, 0, 0), weight, cv2.LINE_AA)
        label_color = (255, 255, 255) if config.state == KeyState.PRESSED else (25, 25, 30)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    label_color, weight, cv2.LINE_AA)

        if config.key != " ":
            dot_x, dot_y = x2 - 10, y1 + 8
            cv2.circle(img, (dot_x, dot_y), 5, finger_color, -1)
            cv2.circle(img, (dot_x, dot_y), 5, (255, 255, 255), 1)

    def _draw_glass_key(self, img, x1, y1, x2, y2, r, finger_color,
                         state, press_progress, confidence, is_target):
        kw, kh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        tint = 0.06
        base = (
            int(255 * (1 - tint) + finger_color[2] * tint),
            int(255 * (1 - tint) + finger_color[1] * tint),
            int(255 * (1 - tint) + finger_color[0] * tint),
        )

        if state == KeyState.HOVER:
            base = tuple(min(255, c + 12) for c in base)
        elif state == KeyState.PRESSED:
            base = tuple(min(255, c + 25) for c in base)
        elif is_target:
            base = tuple(min(255, c + 18) for c in base)

        self._rounded_rect_filled(img, x1, y1, x2, y2, r, base)

        highlight_h = max(6, int(kh * 0.35))
        for i in range(highlight_h):
            y = y1 + i
            if y >= y2 - r: break
            alpha = (1.0 - (i / highlight_h) ** 1.5) * 0.55
            if press_progress > 0:
                alpha *= 1.3
            color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
            left = x1 + r if y > y1 + r else x1 + int(np.sqrt(max(0, r*r - (y - y1 - r)**2)))
            right = x2 - r if y > y1 + r else x2 - int(np.sqrt(max(0, r*r - (y - y1 - r)**2)))
            cv2.line(img, (left, y), (right, y), color, 1)

        cv2.line(img, (x1 + r, y1 + 1), (x2 - r, y1 + 1), (255, 255, 255), 1)

        if is_target:
            pulse = 0.5 + 0.5 * np.sin(self._frame_count * 0.15)
            border_color = (100, 190, 255)
            thickness = 1 + int(pulse * 1.5)
        elif state == KeyState.PRESSED:
            border_color = (160, 210, 255)
            thickness = 2
        else:
            border_color = (170, 170, 180)
            thickness = 1

        self._rounded_rect_outline(img, x1, y1, x2, y2, r, border_color, thickness)

        if press_progress > 0.1:
            max_r = int(max(kw, kh) * 0.55)
            ripple_r = int(max_r * press_progress * (0.5 + 0.5 * confidence))
            alpha = (1.0 - press_progress) * 0.5
            color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
            cv2.circle(img, (cx, cy), ripple_r, color, 1)

    def _draw_finger_guides(self, img, finger_states: Dict, target_key: str, w: int, h: int):
        tc = self.key_configs[target_key]
        tx, ty = int(tc.x * w), int(tc.y * h)
        target_finger = tc.finger
        target_color = FINGER_COLORS.get(target_finger, (100, 200, 255))

        for fname, fstate in finger_states.items():
            if not fstate.visible:
                continue
            fx, fy = int(fstate.x * w), int(fstate.y * h)
            is_owner = fname == target_finger
            color = fstate.color if is_owner else (120, 120, 130)

            self._draw_dashed_line(img, (fx, fy), (tx, ty), color, 12, 8, 2 if is_owner else 1)

            cv2.circle(img, (fx, fy), 10, color, -1)
            cv2.circle(img, (fx, fy), 10, (255, 255, 255), 2)
            cv2.circle(img, (fx, fy), 3, (30, 30, 35), -1)

        cv2.circle(img, (tx, ty), max(w, h) // 18, target_color, 2)
        cv2.circle(img, (tx, ty), max(w, h) // 16, (255, 255, 255), 1)

    def _draw_fingertips(self, img, finger_states: Dict, w: int, h: int):
        for fname, fstate in finger_states.items():
            if not fstate.visible or len(fstate.trail) < 2:
                continue
            color = fstate.color
            trail_pts = [(int(x * w), int(y * h)) for x, y in fstate.trail[-8:]]
            for i in range(1, len(trail_pts)):
                alpha = (i / len(trail_pts)) * 0.25
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(img, trail_pts[i-1], trail_pts[i], trail_color, 1, cv2.LINE_AA)

    def _draw_skeleton(self, img, landmarks, w: int, h: int):
        if not landmarks:
            return
        lm = landmarks
        connections = [
            (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17),(0,17)
        ]
        for start, end in connections:
            s, e = lm[start], lm[end]
            sx, sy = int(s.x * w), int(s.y * h)
            ex, ey = int(e.x * w), int(e.y * h)
            avg_z = (getattr(s, 'z', 0) + getattr(e, 'z', 0)) / 2
            alpha = max(0.15, min(0.4, 0.25 + avg_z * 3))
            color = (int(100 * alpha), int(180 * alpha), int(255 * alpha))
            cv2.line(img, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)

    def _rounded_rect_filled(self, img, x1, y1, x2, y2, r, color):
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)

    def _rounded_rect_outline(self, img, x1, y1, x2, y2, r, color, thickness):
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def _draw_dashed_line(self, img, pt1, pt2, color, dash_len, gap_len, thickness):
        x1, y1 = pt1; x2, y2 = pt2
        dx, dy = x2 - x1, y2 - y1
        dist = (dx*dx + dy*dy)**0.5
        if dist == 0: return
        ux, uy = dx / dist, dy / dist
        pos = 0.0
        while pos < dist:
            sx = int(x1 + ux * pos)
            sy = int(y1 + uy * pos)
            ep = min(pos + dash_len, dist)
            ex = int(x1 + ux * ep)
            ey = int(y1 + uy * ep)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            pos += dash_len + gap_len