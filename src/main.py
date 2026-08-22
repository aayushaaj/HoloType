# -*- coding: utf-8 -*-
"""
main.py - HoloType: Camera-only QWERTY typing with premium glassmorphism UI.

Pipeline: Webcam -> HandTracker -> Smoothing -> TapDetector -> Calibration -> Decoder -> GlassmorphismRenderer

Run modes:
  python -m src.main --calibrate     Guided calibration (single-finger or touch-typing)
  python -m src.main --type          Live typing with saved calibration
  python -m src.main --benchmark     Accuracy benchmark (stub)
"""

import argparse
import time
import os
import sys
import cv2
import logging

sys.path.insert(0, os.path.dirname(__file__))

from hand_tracker import HandTracker
from tap_detector import VelocityTapDetector
from calibration import MultiFingerCalibrator, nearest_key, CALIBRATION_PHRASES
from decoder import NoisyChannelDecoder
from visuals import GlassmorphismRenderer, FingerVisualState, KeyState, FINGER_COLORS, FINGER_COLUMNS
from smoothing import FingertipKalmanFilter, MultiAxisOneEuroFilter
from config import AppConfig, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_calibration(config: AppConfig):
    """Enhanced guided calibration with multi-finger support and premium visuals."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam. Check camera permissions.")
        return

    tracker = HandTracker(
        max_hands=config.hand_tracking.max_hands,
        detection_confidence=config.hand_tracking.detection_confidence,
        tracking_confidence=config.hand_tracking.tracking_confidence,
    )
    tap_detector = VelocityTapDetector(
        history_len=config.tap_detection.history_len,
        down_vel_thresh=config.tap_detection.down_vel_thresh,
        up_vel_thresh=config.tap_detection.up_vel_thresh,
        refractory=config.tap_detection.refractory,
    )

    # Smoothing
    if config.smoothing.enabled:
        if config.smoothing.method == "kalman":
            smoother = FingertipKalmanFilter(
                process_noise=config.smoothing.kalman_process_noise,
                measurement_noise=config.smoothing.kalman_measurement_noise,
            )
        else:
            smoother = MultiAxisOneEuroFilter(
                min_cutoff=config.smoothing.one_euro_min_cutoff,
                beta=config.smoothing.one_euro_beta,
            )
    else:
        smoother = None

    calibrator = MultiFingerCalibrator(mode=config.calibration.mode)
    phrase = config.calibration.phrase or CALIBRATION_PHRASES[config.calibration.mode]
    calibrator.phrase = phrase

    renderer = None
    finger_trails = {f: [] for f in FINGER_COLORS.keys()}
    max_trail_len = 12

    print(f"\n{'='*60}")
    print(f"  CALIBRATION MODE: {config.calibration.mode.upper()}")
    print(f"  Phrase: {phrase}")
    print(f"  Hold hand ~30-40cm from camera.")
    print(f"  Tap each letter with the indicated finger.")
    print(f"  Keys: 'r' = redo current, 'q' = quit")
    print(f"{'='*60}\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if renderer is None:
            renderer = GlassmorphismRenderer(w, h)

        hand_frames = tracker.process(frame)

        target = calibrator.get_next_target()
        target_key = target[0] if target else None
        target_finger = target[1] if target else None

        finger_states = {}
        for hf in hand_frames:
            for finger_name in FINGER_COLORS.keys():
                if finger_name not in hf.fingertips_norm:
                    continue

                tx, ty, tz = hf.fingertips_norm[finger_name]

                if smoother:
                    if isinstance(smoother, FingertipKalmanFilter):
                        tx, ty, tz = smoother.update(finger_name, tx, ty, tz, hf.timestamp)
                    else:
                        tx, ty, tz = smoother.update(finger_name, tx, ty, tz, hf.timestamp)

                pip_x, pip_y, pip_z = hf.pip_norm.get(finger_name, (tx, ty, tz))

                tap = tap_detector.update(
                    finger_name, tx, ty, tz, pip_x, pip_y, pip_z, hf.timestamp
                )

                if tap and target_key and finger_name == target_finger:
                    calibrator.record_sample(
                        target_key, finger_name, tap.x, tap.y, tap.z, tap.confidence
                    )
                    print(f"  ✓ recorded '{target_key}' with {finger_name} (conf={tap.confidence:.2f})")
                    renderer.update_key_state(target_key, KeyState.PRESSED, 1.0, tap.confidence)

                finger_trails[finger_name].append((tx, ty))
                if len(finger_trails[finger_name]) > max_trail_len:
                    finger_trails[finger_name].pop(0)

                finger_states[finger_name] = FingerVisualState(
                    name=finger_name,
                    x=tx, y=ty, z=tz,
                    color=FINGER_COLORS[finger_name],
                    visible=True,
                    trail=finger_trails[finger_name],
                )

        # Render with premium glassmorphism
        frame = renderer.render(
            frame,
            finger_states=finger_states if config.visual.show_finger_trails else None,
            target_key=target_key,
            show_skeleton=config.visual.show_skeleton,
            hand_landmarks=[hf.raw_landmarks for hf in hand_frames] if config.visual.show_skeleton else None,
        )

        # Progress UI
        progress = f"{calibrator.current_key_idx}/{len(phrase)}"
        cv2.putText(frame, f"TAP: '{target_key}'  ({progress})", (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 220, 255), 2)

        metrics = calibrator.get_quality_metrics()
        cv2.putText(frame, f"Samples: {metrics['total_samples']}", (20, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 190), 1)

        cv2.imshow("AIR typer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r") and calibrator.current_key_idx > 0:
            calibrator.current_key_idx -= 1
            if calibrator.current_finger:
                calib = calibrator.finger_calibrations[calibrator.current_finger]
                if calib.samples:
                    calib.samples.pop()
            print(f"  ↩ reset to key {calibrator.current_key_idx}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    os.makedirs(os.path.dirname(config.calibration_path), exist_ok=True)
    calibrator.save(config.calibration_path)
    print(f"\n✓ Calibration saved to {config.calibration_path}")

    metrics = calibrator.get_quality_metrics()
    print("\nCalibration Quality Report:")
    for finger, m in metrics.items():
        if isinstance(m, dict):
            print(f"  {finger:12s}: {m['n_samples']:2d} samples, {m['unique_keys']:2d} keys, avg_conf={m['avg_confidence']:.2f}")


# Map visual finger names to calibration finger names
_VISUAL_TO_CALIB = {
    "pinky": "index",   # fallback to index if not calibrated
    "ring": "index",
    "middle": "index",
    "index": "index",
    "thumb": "thumb",
}

def run_typing(config: AppConfig):
    """Live typing mode with full decoder and premium visuals."""
    if not os.path.exists(config.calibration_path):
        print("No calibration profile found. Run --calibrate first.")
        return

    cal_data = MultiFingerCalibrator.load(config.calibration_path)
    cal_mode = cal_data.get('mode', 'unknown')
    if "centroids" in cal_data:
        centroids = cal_data["centroids"]
    else:
        centroids = cal_data
    print(f"✓ Loaded calibration for {len(centroids)} keys (mode: {cal_mode})")

    # Determine which fingers to use for tap detection based on calibration mode
    if cal_mode == "single_finger":
        active_fingers = ["index"]  # Only index finger was calibrated
    else:
        active_fingers = list(FINGER_COLORS.keys())  # All fingers for touch_typing

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam.")
        return

    tracker = HandTracker(
        max_hands=config.hand_tracking.max_hands,
        detection_confidence=config.hand_tracking.detection_confidence,
        tracking_confidence=config.hand_tracking.tracking_confidence,
    )
    tap_detector = VelocityTapDetector(
        history_len=config.tap_detection.history_len,
        down_vel_thresh=config.tap_detection.down_vel_thresh,
        up_vel_thresh=config.tap_detection.up_vel_thresh,
        refractory=config.tap_detection.refractory,
        debug=args.debug_tap,
    )

    if config.smoothing.enabled:
        if config.smoothing.method == "kalman":
            smoother = FingertipKalmanFilter(
                process_noise=config.smoothing.kalman_process_noise,
                measurement_noise=config.smoothing.kalman_measurement_noise,
            )
        else:
            smoother = MultiAxisOneEuroFilter(
                min_cutoff=config.smoothing.one_euro_min_cutoff,
                beta=config.smoothing.one_euro_beta,
            )
    else:
        smoother = None

    decoder = NoisyChannelDecoder(vocabulary_path=config.vocabulary_path or None)

    renderer = None
    finger_trails = {f: [] for f in FINGER_COLORS.keys()}
    max_trail_len = 12

    highlight_key = None
    highlight_until = 0.0

    print("\n" + "="*60)
    print("  LIVE TYPING MODE")
    print("  Type naturally in the air. Press 'c' to clear, 'q' to quit.")
    print(f"  Active fingers for tap detection: {active_fingers}")
    print("="*60 + "\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if renderer is None:
            renderer = GlassmorphismRenderer(w, h)

        hand_frames = tracker.process(frame)
        finger_states = {}

        for hf in hand_frames:
            for finger_name in FINGER_COLORS.keys():
                if finger_name not in hf.fingertips_norm:
                    continue

                tx, ty, tz = hf.fingertips_norm[finger_name]

                if smoother:
                    if isinstance(smoother, FingertipKalmanFilter):
                        tx, ty, tz = smoother.update(finger_name, tx, ty, tz, hf.timestamp)
                    else:
                        tx, ty, tz = smoother.update(finger_name, tx, ty, tz, hf.timestamp)

                pip_x, pip_y, pip_z = hf.pip_norm.get(finger_name, (tx, ty, tz))

                # Only detect taps on calibrated fingers
                if finger_name in active_fingers:
                    tap = tap_detector.update(
                        finger_name, tx, ty, tz, pip_x, pip_y, pip_z, hf.timestamp
                    )

                    if tap:
                        # Map visual finger name to calibration finger name
                        calib_finger = _VISUAL_TO_CALIB.get(finger_name, finger_name)
                        key, xy_dist, z_dist = nearest_key(
                            tap.x, tap.y, tap.z, calib_finger, centroids
                        )

                        decoder.feed_key(
                            key=key,
                            finger=calib_finger,
                            xy_dist=xy_dist,
                            z_dist=z_dist,
                            tap_confidence=tap.confidence,
                            timestamp=tap.timestamp,
                        )

                        print(f"  tap: '{key}' (xy={xy_dist:.3f}, z={z_dist:.3f}, conf={tap.confidence:.2f})")

                        highlight_key = key
                        highlight_until = time.time() + 0.35
                        renderer.update_key_state(key, KeyState.PRESSED, 1.0, tap.confidence)

                finger_trails[finger_name].append((tx, ty))
                if len(finger_trails[finger_name]) > max_trail_len:
                    finger_trails[finger_name].pop(0)

                finger_states[finger_name] = FingerVisualState(
                    name=finger_name,
                    x=tx, y=ty, z=tz,
                    color=FINGER_COLORS[finger_name],
                    visible=True,
                    trail=finger_trails[finger_name],
                )

        # Render keyboard
        frame = renderer.render(
            frame,
            finger_states=finger_states if config.visual.show_finger_trails else None,
            show_skeleton=config.visual.show_skeleton,
            hand_landmarks=[hf.raw_landmarks for hf in hand_frames] if config.visual.show_skeleton else None,
        )

        # Decoded text display
        display_text = decoder.decoded_text[-100:] + (" " + decoder.get_raw_buffer() if decoder.get_raw_buffer() else "")
        cv2.putText(frame, display_text, (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (80, 220, 255), 2)

        # Raw buffer
        raw = decoder.get_raw_buffer()
        if raw:
            cv2.putText(frame, f"Raw: {raw}", (20, 88),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (40, 30, 80), 1)

        # Mode indicator
        mode_text = f"Mode: {config.calibration.mode}  |  Decoder: {config.decoder.method}"
        cv2.putText(frame, mode_text, (20, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (120, 120, 130), 1)

        cv2.imshow("AIR typer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            decoder.decoded_text = ""
            decoder.buffer = []

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    final_text = decoder.finalize()
    print("\n" + "="*60)
    print("  FINAL DECODED TEXT")
    print("="*60)
    print(final_text)
    print("="*60)


def run_benchmark(config: AppConfig):
    """Accuracy benchmark with ground truth phrases."""
    print("Benchmark mode - not fully implemented yet.")
    print("Run --calibrate then --type for now.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HoloType - Camera-only QWERTY typing with glassmorphism UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --calibrate                    # Touch-typing calibration
  python -m src.main --calibrate --mode single_finger  # Quick single-finger
  python -m src.main --type                         # Live typing
  python -m src.main --type --show-skeleton         # With hand skeleton debug
  python -m src.main --type --smoothing one_euro    # One-Euro filter
        """
    )
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--type", action="store_true", help="Run live typing")
    parser.add_argument("--benchmark", action="store_true", help="Run accuracy benchmark")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--mode", choices=["single_finger", "touch_typing"],
                       default="touch_typing", help="Calibration mode")
    parser.add_argument("--smoothing", choices=["kalman", "one_euro", "none"],
                       default="kalman", help="Smoothing method")
    parser.add_argument("--decoder", choices=["spellchecker", "noisy_channel"],
                       default="noisy_channel", help="Decoder method")
    parser.add_argument("--show-skeleton", action="store_true", help="Show hand skeleton overlay")
    parser.add_argument("--no-trails", action="store_true", help="Hide finger trails")
    parser.add_argument("--debug-tap", action="store_true", help="Enable tap detection debug output")

    args = parser.parse_args()

    config = AppConfig.load(args.config) if os.path.exists(args.config) else DEFAULT_CONFIG

    config.calibration.mode = args.mode
    config.smoothing.method = args.smoothing if args.smoothing != "none" else "kalman"
    config.smoothing.enabled = args.smoothing != "none"
    config.decoder.method = args.decoder
    config.visual.show_skeleton = args.show_skeleton
    config.visual.show_finger_trails = not args.no_trails

    config.save(args.config)

    if args.calibrate:
        run_calibration(config)
    elif args.type:
        run_typing(config)
    elif args.benchmark:
        run_benchmark(config)
    else:
        parser.print_help()
        print("\nTry: python -m src.main --calibrate  (first run)")
        print("     python -m src.main --type       (after calibration)")