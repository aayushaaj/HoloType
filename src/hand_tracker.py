# -*- coding: utf-8 -*-
"""HandTracker - MediaPipe Hands wrapper returning normalized fingertip positions."""

from dataclasses import dataclass, field
import time
import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from pathlib import Path

FINGERTIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


@dataclass
class HandFrame:
    handedness: str
    timestamp: float
    fingertips_norm: dict
    fingertips_px: dict
    pip_norm: dict
    raw_landmarks: object = field(repr=False, default=None)


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.6, tracking_confidence=0.6):
        if hasattr(mp, "solutions"):
            self._api = "solutions"
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self._api = "tasks"
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_dir = Path(__file__).resolve().parents[1] / "data" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "hand_landmarker.task"
            if not model_path.exists():
                MODEL_URL = (
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                )
                try:
                    urllib.request.urlretrieve(MODEL_URL, model_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to download MediaPipe model: {e}")

            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                running_mode=mp_vision.RunningMode.VIDEO,
            )
            self.hands = mp_vision.HandLandmarker.create_from_options(options)
            self.mp_draw = mp_vision.drawing_utils
            self.mp_hands = mp_vision

    def process(self, frame_bgr) -> list[HandFrame]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = None
        if self._api == "solutions":
            results = self.hands.process(frame_rgb)
        else:
            ts_ms = int(time.time() * 1000)
            mp_image = mp.Image(mp.ImageFormat.SRGB, frame_rgb)
            results = self.hands.detect_for_video(mp_image, ts_ms)

        hand_frames = []
        if self._api == "solutions":
            if not results.multi_hand_landmarks:
                return hand_frames
            ts = time.time()
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                lm = hand_landmarks.landmark
                tips_norm, tips_px, pip_norm = {}, {}, {}
                for finger, idx in FINGERTIP_IDS.items():
                    point = lm[idx]
                    tips_norm[finger] = (point.x, point.y, point.z)
                    tips_px[finger] = (int(point.x * w), int(point.y * h))
                for finger, idx in PIP_IDS.items():
                    point = lm[idx]
                    pip_norm[finger] = (point.x, point.y, point.z)
                hand_frames.append(HandFrame(label, ts, tips_norm, tips_px, pip_norm, hand_landmarks))
            return hand_frames

        if not results.hand_landmarks:
            return hand_frames
        ts = time.time()
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = results.handedness[idx][0] if results.handedness else None
            label = handedness.category_name if handedness is not None else "Unknown"
            lm = hand_landmarks
            tips_norm, tips_px, pip_norm = {}, {}, {}
            for finger, idx_id in FINGERTIP_IDS.items():
                point = lm[idx_id]
                tips_norm[finger] = (point.x, point.y, getattr(point, 'z', 0.0))
                tips_px[finger] = (int(point.x * w), int(point.y * h))
            for finger, idx_id in PIP_IDS.items():
                point = lm[idx_id]
                pip_norm[finger] = (point.x, point.y, getattr(point, 'z', 0.0))
            hand_frames.append(HandFrame(label, ts, tips_norm, tips_px, pip_norm, hand_landmarks))
        return hand_frames

    def draw(self, frame_bgr, hand_frames_raw):
        h, w = frame_bgr.shape[:2]
        for hand_landmarks in hand_frames_raw:
            try:
                lm_list = list(hand_landmarks.landmark)
            except Exception:
                lm_list = list(hand_landmarks)
            for lm in lm_list:
                x_px, y_px = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (x_px, y_px), 4, (0, 255, 0), -1)
            try:
                connections = self.mp_hands.HAND_CONNECTIONS
                for c in connections:
                    start = c[0] if isinstance(c, (list, tuple)) else c.start
                    end = c[1] if isinstance(c, (list, tuple)) else c.end
                    s, e = lm_list[start], lm_list[end]
                    cv2.line(frame_bgr, (int(s.x * w), int(s.y * h)), (int(e.x * w), int(e.y * h)), (255, 0, 0), 2)
            except Exception:
                pass
        return frame_bgr

    def close(self):
        self.hands.close()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    print("Press 'q' to quit.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hand_frames = tracker.process(frame)
        for hf in hand_frames:
            for finger, (px, py) in hf.fingertips_px.items():
                cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(frame, finger[:2], (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("Hand Tracker Sanity Check", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()