"""GlassmorphismRenderer — Backlit deep-glass iOS keyboard (60 FPS optimized).

Visual language matches an iOS "frosted alarm panel": deep tinted glass,
soft outer bloom/glow on active elements, glowing LCD-style labels, and
pill-shaped rounded panels.
"""

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


# BGR tuples — glow accent colors per finger (used for tags, rings, ripples)
FINGER_COLORS = {
    "pinky": (110, 84, 255),    # coral / red-pink
    "ring": (54, 176, 255),     # amber / orange
    "middle": (110, 210, 90),   # mint green
    "index": (255, 200, 90),    # cyan-blue glow (matches reference LCD look)
    "thumb": (255, 92, 199),    # violet / magenta
}

FINGER_COLUMNS = {
    "pinky": list("qaz") + ["p", ";", "/"],
    "ring": list("wsx") + ["o", "l"],
    "middle": list("edc") + ["i", "k", ","],
    "index": list("rfvtgb") + list("yhnujm"),
    "thumb": [" "],
}


def _mix(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


def _clampc(c):
    return tuple(int(np.clip(v, 0, 255)) for v in c)


class GlassmorphismRenderer:
    """Deep backlit-glass keyboard: tinted glass panels, glow blooms, LCD-style labels."""

    # Deep blue glass panel tone (BGR) — mirrors the alarm-clock reference art
    GLASS_DEEP = (150, 95, 45)
    GLASS_DEEP_LIGHT = (205, 150, 85)
    GLOW_CYAN = (255, 230, 170)  # soft cyan-white glow used for active segments/labels
    INK = (235, 240, 245)

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.key_radius = 20
        self._init_key_layout()
        self._frame_count = 0
        self._blur_k = max(15, (min(frame_w, frame_h) // 40) | 1)

    def _init_key_layout(self):
        self.key_configs: Dict[str, KeyVisualConfig] = {}

        # Physical-keyboard stagger offsets (fraction of a key width)
        rows = [
            (list("qwertyuiop"), 0.36, 0.0),
            (list("asdfghjkl"), 0.445, 0.55),
            (list("zxcvbnm"), 0.53, 1.35),
        ]
        pad = 0.045
        key_w = (1.0 - 2 * pad) / 10.2
        gap = key_w * 0.16

        for row_idx, (row, y, stagger_units) in enumerate(rows):
            start_x = pad + stagger_units * (key_w + gap)
            for i, key in enumerate(row):
                cx = start_x + i * (key_w + gap) + key_w / 2
                finger = self._get_finger_for_key(key)
                self.key_configs[key] = KeyVisualConfig(
                    key=key, x=cx, y=y, finger=finger,
                    width=key_w * 0.94,
                    height=0.082
                )

        # Space bar — pill shaped, like the "Snooze" pill in the reference
        self.key_configs[" "] = KeyVisualConfig(
            key=" ", x=0.5, y=0.685, finger="thumb", width=0.30, height=0.072
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

    # ------------------------------------------------------------------ #
    #  Main render pass
    # ------------------------------------------------------------------ #

    def render(self, frame: np.ndarray,
               finger_states: Optional[Dict[str, FingerVisualState]] = None,
               target_key: Optional[str] = None,
               show_skeleton: bool = False,
               hand_landmarks: Optional[List] = None) -> np.ndarray:
        self._frame_count += 1
        h, w = frame.shape[:2]

        # Frosted backdrop sampled from the live frame — gives keys real depth/refraction
        frosted = cv2.GaussianBlur(frame, (self._blur_k, self._blur_k), 0)
        frosted = cv2.addWeighted(frosted, 0.55, np.full_like(frosted, 40), 0.45, 0)

        overlay = frame.copy()
        shadow_layer = np.zeros_like(frame)
        glow_layer = np.zeros_like(frame)

        # Soft drop shadows under every key
        for config in self.key_configs.values():
            self._stamp_shadow(shadow_layer, config, w, h)
        shadow_layer = cv2.GaussianBlur(shadow_layer, (0, 0), sigmaX=10)
        cv2.addWeighted(shadow_layer, 0.6, overlay, 1.0, 0, overlay)

        # Bloom pass: stamp bright shapes for active/target/pressed keys, blur once,
        # then screen-blend on top for the LCD-glow look
        for key, config in self.key_configs.items():
            if config.state in (KeyState.PRESSED, KeyState.TARGET) or key == target_key:
                self._stamp_glow(glow_layer, config, w, h)
        glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=14)
        overlay = cv2.add(overlay, (glow_layer.astype(np.float32) * 0.55).astype(np.uint8))

        # Glass keys
        for key, config in self.key_configs.items():
            self._draw_key(overlay, frosted, config, w, h, target_key)

        if finger_states and target_key and target_key in self.key_configs:
            self._draw_finger_guides(overlay, finger_states, target_key, w, h)

        if finger_states:
            self._draw_fingertips(overlay, finger_states, w, h)

        if show_skeleton and hand_landmarks:
            self._draw_skeleton(overlay, hand_landmarks, w, h)

        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        return frame

    # ------------------------------------------------------------------ #
    #  Key drawing
    # ------------------------------------------------------------------ #

    def _stamp_shadow(self, shadow_img, config: KeyVisualConfig, w: int, h: int):
        cx = int(config.x * w)
        cy = int(config.y * h) + 5
        kw = int(config.width * w)
        kh = int(config.height * h)
        x1, y1 = cx - kw // 2, cy - kh // 2
        x2, y2 = cx + kw // 2, cy + kh // 2
        r = kh // 2 if config.key == " " else self.key_radius
        self._rounded_rect_filled(shadow_img, x1, y1, x2, y2, r, (30, 20, 10))

    def _stamp_glow(self, glow_img, config: KeyVisualConfig, w: int, h: int):
        cx = int(config.x * w)
        cy = int(config.y * h)
        kw = int(config.width * w * 1.15)
        kh = int(config.height * h * 1.25)
        x1, y1 = cx - kw // 2, cy - kh // 2
        x2, y2 = cx + kw // 2, cy + kh // 2
        r = kh // 2 if config.key == " " else self.key_radius + 6
        finger_color = FINGER_COLORS.get(config.finger, self.GLOW_CYAN)
        glow_color = _clampc(_mix(finger_color, (255, 255, 255), 0.5))
        self._rounded_rect_filled(glow_img, x1, y1, x2, y2, r, glow_color)

    def _draw_key(self, img: np.ndarray, frosted: np.ndarray, config: KeyVisualConfig,
                  w: int, h: int, target_key: str = None):
        cx = int(config.x * w)
        cy = int(config.y * h)
        kw = int(config.width * w)
        kh = int(config.height * h)
        x1, y1 = cx - kw // 2, cy - kh // 2
        x2, y2 = cx + kw // 2, cy + kh // 2
        is_pill = config.key == " "
        r = kh // 2 if is_pill else self.key_radius

        # Press = slight scale-down, like a real key sinking into the glass
        if config.press_progress > 0:
            scale = 1.0 - 0.05 * config.press_progress
            nkw, nkh = int(kw * scale), int(kh * scale)
            x1, x2 = cx - nkw // 2, cx + nkw // 2
            y1, y2 = cy - nkh // 2, cy + nkh // 2

        finger_color = FINGER_COLORS.get(config.finger, (180, 180, 180))
        is_target = target_key is not None and config.key == target_key

        self._draw_glass_key(img, frosted, x1, y1, x2, y2, r, finger_color,
                              config.state, config.press_progress, config.confidence, is_target)

        # --- Center label, rendered LCD-style with a soft glow underlay ---
        label = config.key.upper() if config.key != " " else "SPACE"
        font_scale = 0.7 if config.key != " " else 0.55
        weight = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, weight)
        tx = cx - tw // 2
        ty = cy + th // 2 - 1

        active = config.state in (KeyState.PRESSED, KeyState.TARGET) or is_target
        label_color = _clampc(_mix(finger_color, (255, 255, 255), 0.6)) if active else self.INK

        if active:
            # Blurred glow pass behind the sharp label for a neon/LCD feel
            glow_canvas = np.zeros_like(img)
            cv2.putText(glow_canvas, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                        label_color, weight + 3, cv2.LINE_AA)
            glow_canvas = cv2.GaussianBlur(glow_canvas, (0, 0), sigmaX=5)
            img[:] = cv2.add(img, (glow_canvas.astype(np.float32) * 0.6).astype(np.uint8))

        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    label_color, weight, cv2.LINE_AA)

        # --- Colored finger-tag chip, top-left corner ---
        if config.key != " ":
            tag = config.finger[0].upper()
            tag_scale = 0.32
            (ttw, tth), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, tag_scale, 1)
            tag_x = x1 + 8
            tag_y = y1 + tth + 7
            chip_r = max(ttw, tth) // 2 + 4
            chip_color = _clampc(_mix(finger_color, (20, 15, 10), 0.15))
            cv2.circle(img, (tag_x + ttw // 2, tag_y - tth // 2), chip_r, chip_color, -1, cv2.LINE_AA)
            cv2.circle(img, (tag_x + ttw // 2, tag_y - tth // 2), chip_r, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, tag, (tag_x, tag_y), cv2.FONT_HERSHEY_DUPLEX, tag_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_glass_key(self, img, frosted, x1, y1, x2, y2, r, finger_color,
                         state, press_progress, confidence, is_target):
        """Deep tinted glass key: dark frosted fill + top sheen + bright rim + inner glow ring."""
        kw, kh = x2 - x1, y2 - y1
        if kw <= 2 or kh <= 2:
            return
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        H, W = img.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(W, x2), min(H, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return

        mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
        self._rounded_rect_filled(mask, 0, 0, cx2 - cx1, cy2 - cy1, r,
                                   255, offset=(-cx1 + x1, -cy1 + y1))

        # --- 1. Deep glass base, blended with the frosted backdrop for refraction ---
        backdrop = frosted[cy1:cy2, cx1:cx2].astype(np.float32)
        tint = 0.22
        tint_color = np.array(finger_color, dtype=np.float32)
        base_deep = np.array(self.GLASS_DEEP, dtype=np.float32)
        glass_base = base_deep * (1 - tint) + tint_color * tint

        state_boost = {
            KeyState.HOVER: 14, KeyState.PRESSED: 34, KeyState.TARGET: 26,
        }.get(state, 0)
        if is_target:
            state_boost = max(state_boost, 26)
        glass_base = np.clip(glass_base + state_boost, 0, 255)

        fill = backdrop * 0.35 + glass_base * 0.65
        region = img[cy1:cy2, cx1:cx2].astype(np.float32)
        m3 = (mask[:, :, None].astype(np.float32) / 255.0)
        blended = region * (1 - m3 * 0.92) + fill * (m3 * 0.92)
        img[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)

        # --- 2. Soft top sheen ---
        highlight_h = max(6, int(kh * 0.5))
        sheen = np.zeros((cy2 - cy1, cx2 - cx1, 3), dtype=np.float32)
        for i in range(min(highlight_h, cy2 - cy1)):
            alpha = (1.0 - (i / highlight_h) ** 1.7) * 0.5
            sheen[i, :, :] = 255 * alpha
        sheen_mask = (mask[:, :, None].astype(np.float32) / 255.0)
        region2 = img[cy1:cy2, cx1:cx2].astype(np.float32)
        region2 = np.clip(region2 + sheen * sheen_mask * 0.30, 0, 255)
        img[cy1:cy2, cx1:cx2] = region2.astype(np.uint8)

        # --- 3. Bright crisp rim-light on top/left inner edge ---
        rim_color = _clampc(_mix(self.GLASS_DEEP_LIGHT, (255, 255, 255), 0.5))
        cv2.line(img, (x1 + r, y1 + 1), (x2 - r, y1 + 1), rim_color, 1, cv2.LINE_AA)
        cv2.line(img, (x1 + 1, y1 + r), (x1 + 1, y2 - r), rim_color, 1, cv2.LINE_AA)

        # --- 4. Outer border ---
        if is_target:
            pulse = 0.5 + 0.5 * np.sin(self._frame_count * 0.15)
            border_color = _clampc(_mix((160, 220, 255), (255, 255, 255), pulse * 0.4))
            thickness = 2 + int(pulse * 2)
        elif state == KeyState.PRESSED:
            border_color = _clampc(_mix(finger_color, (255, 255, 255), 0.45))
            thickness = 2
        elif state == KeyState.HOVER:
            border_color = _clampc(_mix(finger_color, (255, 255, 255), 0.6))
            thickness = 2
        else:
            border_color = _clampc(_mix(self.GLASS_DEEP_LIGHT, (255, 255, 255), 0.25))
            thickness = 1

        self._rounded_rect_outline(img, x1, y1, x2, y2, r, border_color, thickness)

        # --- 5. Press ripple, colored by finger ---
        if press_progress > 0.08:
            max_r = int(max(kw, kh) * 0.6)
            ripple_r = int(max_r * press_progress * (0.5 + 0.5 * confidence))
            alpha = (1.0 - press_progress) * 0.6
            color = _clampc(tuple(c * alpha for c in finger_color))
            cv2.circle(img, (cx, cy), ripple_r, color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Finger guides / trails / skeleton
    # ------------------------------------------------------------------ #

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
            color = fstate.color if is_owner else (150, 150, 158)

            self._draw_dashed_line(img, (fx, fy), (tx, ty), color, 12, 8, 2 if is_owner else 1)

            cv2.circle(img, (fx, fy), 11, color, -1, cv2.LINE_AA)
            cv2.circle(img, (fx, fy), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (fx, fy), 3, (30, 30, 35), -1, cv2.LINE_AA)

        cv2.circle(img, (tx, ty), max(w, h) // 18, target_color, 2, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), max(w, h) // 16, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_fingertips(self, img, finger_states: Dict, w: int, h: int):
        for fname, fstate in finger_states.items():
            if not fstate.visible or len(fstate.trail) < 2:
                continue
            color = fstate.color
            trail_pts = [(int(x * w), int(y * h)) for x, y in fstate.trail[-8:]]
            for i in range(1, len(trail_pts)):
                alpha = (i / len(trail_pts)) * 0.3
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(img, trail_pts[i - 1], trail_pts[i], trail_color, 2, cv2.LINE_AA)

    def _draw_skeleton(self, img, landmarks, w: int, h: int):
        if not landmarks:
            return
        lm = landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
        for start, end in connections:
            s, e = lm[start], lm[end]
            sx, sy = int(s.x * w), int(s.y * h)
            ex, ey = int(e.x * w), int(e.y * h)
            avg_z = (getattr(s, 'z', 0) + getattr(e, 'z', 0)) / 2
            alpha = max(0.15, min(0.4, 0.25 + avg_z * 3))
            color = (int(100 * alpha), int(180 * alpha), int(255 * alpha))
            cv2.line(img, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Drawing primitives
    # ------------------------------------------------------------------ #

    def _rounded_rect_filled(self, img, x1, y1, x2, y2, r, color, offset: Tuple[int, int] = (0, 0)):
        ox, oy = offset
        x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
        r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
        if r < 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
            return
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)

    def _rounded_rect_outline(self, img, x1, y1, x2, y2, r, color, thickness):
        r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

    def _draw_dashed_line(self, img, pt1, pt2, color, dash_len, gap_len, thickness):
        x1, y1 = pt1
        x2, y2 = pt2
        dx, dy = x2 - x1, y2 - y1
        dist = (dx * dx + dy * dy) ** 0.5
        if dist == 0:
            return
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