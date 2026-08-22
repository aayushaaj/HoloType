# -*- coding: utf-8 -*-
"""HoloType CLI and runtime orchestration."""

import argparse
import logging
import os
import time

from .calibration import CALIBRATION_PHRASES, MultiFingerCalibrator, nearest_key
from .config import AppConfig, DEFAULT_CONFIG
from .runtime import (
    active_fingers_for_mode,
    calibration_is_usable,
    canonical_finger,
    configured_max_hands,
    create_decoder,
    create_smoother,
    create_tap_detector,
    detection_finger_for_profile,
    display_finger_name,
    effective_mode_for_calibration,
    feed_decoder,
    load_jsonl,
    metrics_to_dict,
    score_text,
    visual_finger_name,
    SessionLogger,
)
from .visuals import FINGER_COLORS, FingerVisualState, GlassmorphismRenderer, KeyState


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
NAVY_BLUE = (128, 0, 0)


def _cv2():
    import cv2

    return cv2


def _open_camera(index: int = 0):
    cv2 = _cv2()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Cannot open webcam. Check camera permissions.")
        return None
    return cap


def _create_tracker(config: AppConfig, mode: str):
    from .hand_tracker import HandTracker

    return HandTracker(
        max_hands=configured_max_hands(mode, config.hand_tracking.max_hands),
        detection_confidence=config.hand_tracking.detection_confidence,
        tracking_confidence=config.hand_tracking.tracking_confidence,
    )


def _update_finger_state(finger_trails, canonical_name, x, y, z):
    visual_name = visual_finger_name(canonical_name)
    finger_trails.setdefault(canonical_name, []).append((x, y))
    del finger_trails[canonical_name][:-12]
    return FingerVisualState(
        name=canonical_name,
        x=x,
        y=y,
        z=z,
        color=FINGER_COLORS.get(visual_name, (180, 180, 180)),
        visible=True,
        trail=finger_trails[canonical_name],
    )


def _matches_target_finger(detected_finger: str, target_finger: str) -> bool:
    return (
        detected_finger == target_finger
        or visual_finger_name(detected_finger) == visual_finger_name(target_finger)
    )


def _print_quality_report(metrics: dict):
    print("\nCalibration Quality Report:")
    print(f"  completion: {metrics.get('completion', 0.0):.0%}")
    print(f"  samples:    {metrics.get('total_samples', 0)}")
    for finger, values in metrics.items():
        if not isinstance(values, dict) or values.get("n_samples", 0) == 0:
            continue
        print(
            f"  {display_finger_name(finger):12s}: "
            f"{values['n_samples']:2d} samples, "
            f"{values['unique_keys']:2d} keys, "
            f"avg_conf={values['avg_confidence']:.2f}"
        )


def run_setup(config: AppConfig, args) -> int:
    """Fast first-run sanity check without writing calibration."""
    cv2 = _cv2()
    cap = _open_camera(args.camera)
    if cap is None:
        return 1
    tracker = _create_tracker(config, config.calibration.mode)
    print("\nCamera check: show one or both hands for 5 seconds. Press q to stop early.")
    start = time.time()
    seen_frames = 0
    try:
        while time.time() - start < 5:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hand_frames = tracker.process(frame)
            seen_frames += int(bool(hand_frames))
            cv2.putText(
                frame,
                f"Hands visible: {len(hand_frames)}",
                (20, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (80, 220, 255),
                2,
            )
            cv2.imshow("HoloType Setup", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

    if seen_frames == 0:
        print("No hands detected. Improve lighting, move closer, or check camera permissions.")
        return 1
    print("Camera and hand tracking look ready. Run calibration next.")
    return 0


def run_calibration(config: AppConfig, args) -> int:
    """Guided calibration with handed finger tracking and save gating."""
    cv2 = _cv2()
    mode = config.calibration.mode
    cap = _open_camera(args.camera)
    if cap is None:
        return 1

    tracker = _create_tracker(config, mode)
    tap_detector = create_tap_detector(config, debug=args.debug_tap)
    smoother = create_smoother(config)
    calibrator = MultiFingerCalibrator(mode=mode)
    calibrator.phrase = config.calibration.phrase or CALIBRATION_PHRASES[mode]
    logger_session = SessionLogger(config.log_dir, "calibration", enabled=args.log_session)

    renderer = None
    finger_trails = {}
    saved = False

    print(f"\n{'=' * 60}")
    print(f"  CALIBRATION MODE: {mode.upper()}")
    print(f"  Phrase: {calibrator.phrase}")
    print("  Hold hand ~30-40cm from camera.")
    print("  Tap each letter with the indicated finger.")
    print("  Keys: r = redo current, s = save if usable, q = quit")
    print(f"{'=' * 60}\n")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if renderer is None:
                renderer = GlassmorphismRenderer(w, h)

            target = calibrator.get_next_target()
            target_key = target[0] if target else None
            target_finger = target[1] if target else None
            finger_states = {}
            hand_frames = tracker.process(frame)

            for hf in hand_frames:
                for raw_finger in FINGER_COLORS.keys():
                    if raw_finger not in hf.fingertips_norm:
                        continue
                    finger = canonical_finger(hf.handedness, raw_finger, mode)
                    tx, ty, tz = hf.fingertips_norm[raw_finger]
                    if smoother:
                        tx, ty, tz = smoother.update(finger, tx, ty, tz, hf.timestamp)
                    pip_x, pip_y, pip_z = hf.pip_norm.get(raw_finger, (tx, ty, tz))
                    tap = tap_detector.update(finger, tx, ty, tz, pip_x, pip_y, pip_z, hf.timestamp)

                    if tap and target_key and _matches_target_finger(finger, target_finger):
                        calibrator.record_sample(
                            target_key, target_finger, tap.x, tap.y, tap.z, tap.confidence
                        )
                        logger_session.write(
                            "calibration_tap",
                            {
                                "key": target_key,
                                "finger": target_finger,
                                "x": tap.x,
                                "y": tap.y,
                                "z": tap.z,
                                "confidence": tap.confidence,
                            },
                        )
                        print(
                            f"  recorded '{target_key}' with "
                            f"{display_finger_name(target_finger)} (conf={tap.confidence:.2f})"
                        )
                        renderer.update_key_state(target_key, KeyState.PRESSED, 1.0, tap.confidence)

                    finger_states[finger] = _update_finger_state(finger_trails, finger, tx, ty, tz)

            frame = renderer.render(
                frame,
                finger_states=finger_states if config.visual.show_finger_trails else None,
                target_key=target_key,
                show_skeleton=config.visual.show_skeleton,
                hand_landmarks=[hf.raw_landmarks for hf in hand_frames] if config.visual.show_skeleton else None,
            )
            progress = f"{calibrator.current_key_idx}/{len(calibrator.phrase)}"
            prompt = "Complete. Press s to save." if target_key is None else (
                f"TAP: '{target_key}' with {display_finger_name(target_finger)} ({progress})"
            )
            cv2.putText(frame, prompt, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.85, NAVY_BLUE, 2)
            cv2.putText(
                frame,
                f"Samples: {calibrator.get_quality_metrics()['total_samples']}",
                (20, 90),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (180, 180, 190),
                1,
            )

            cv2.imshow("HoloType Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and calibrator.current_key_idx > 0:
                calibrator.current_key_idx -= 1
                if calibrator.current_finger:
                    calib = calibrator.finger_calibrations[calibrator.current_finger]
                    if calib.samples:
                        calib.samples.pop()
                print(f"  reset to key {calibrator.current_key_idx}")
            if key == ord("s") or target_key is None:
                metrics = calibrator.get_quality_metrics()
                usable, reason = calibration_is_usable(
                    metrics, mode, config.calibration.min_samples_per_key
                )
                _print_quality_report(metrics)
                if usable or args.force_save:
                    calibrator.save(config.calibration_path)
                    logger_session.write("calibration_saved", {"path": config.calibration_path})
                    print(f"\nCalibration saved to {config.calibration_path}")
                    saved = True
                    break
                print(f"\nCalibration not saved: {reason}. Use --force-save to override.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

    if not saved:
        metrics = calibrator.get_quality_metrics()
        _print_quality_report(metrics)
        logger_session.write("calibration_discarded", {"quality": metrics})
        return 1
    return 0


def run_typing(config: AppConfig, args) -> int:
    """Live typing mode with decoder, session logging, and optional validation metrics."""
    cv2 = _cv2()
    if not os.path.exists(config.calibration_path):
        print("No calibration profile found. Run calibration first.")
        return 1

    cal_data = MultiFingerCalibrator.load(config.calibration_path)
    cal_mode = cal_data.get("mode", config.calibration.mode)
    centroids = cal_data.get("centroids", cal_data)
    if not centroids:
        print("Calibration profile has no centroids. Run calibration again.")
        return 1
    runtime_mode = effective_mode_for_calibration(cal_mode, centroids)

    cap = _open_camera(args.camera)
    if cap is None:
        return 1

    tracker = _create_tracker(config, runtime_mode)
    tap_detector = create_tap_detector(config, debug=args.debug_tap)
    smoother = create_smoother(config)
    decoder = create_decoder(config)
    logger_session = SessionLogger(config.log_dir, "typing", enabled=args.log_session)

    renderer = None
    finger_trails = {}
    active_fingers = active_fingers_for_mode(runtime_mode)
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  LIVE TYPING MODE")
    print("  Type naturally in the air. Press c to clear, q to quit.")
    if args.expected_text:
        print(f"  Validation target: {args.expected_text}")
    print(f"  Active fingers: {', '.join(display_finger_name(f) for f in sorted(active_fingers))}")
    print("=" * 60 + "\n")
    logger_session.write(
        "typing_started",
        {"calibration_mode": runtime_mode, "expected_text": args.expected_text or ""},
    )

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if renderer is None:
                renderer = GlassmorphismRenderer(w, h)

            finger_states = {}
            hand_frames = tracker.process(frame)
            for hf in hand_frames:
                for raw_finger in FINGER_COLORS.keys():
                    if raw_finger not in hf.fingertips_norm:
                        continue
                    finger = detection_finger_for_profile(
                        hf.handedness, raw_finger, runtime_mode, centroids
                    )
                    tx, ty, tz = hf.fingertips_norm[raw_finger]
                    if smoother:
                        tx, ty, tz = smoother.update(finger, tx, ty, tz, hf.timestamp)
                    pip_x, pip_y, pip_z = hf.pip_norm.get(raw_finger, (tx, ty, tz))

                    if finger in active_fingers:
                        tap = tap_detector.update(
                            finger, tx, ty, tz, pip_x, pip_y, pip_z, hf.timestamp
                        )
                        if tap:
                            key, xy_dist, z_dist = nearest_key(tap.x, tap.y, tap.z, finger, centroids)
                            feed_decoder(
                                decoder, key, finger, xy_dist, z_dist, tap.confidence, tap.timestamp
                            )
                            logger_session.write(
                                "tap",
                                {
                                    "key": key,
                                    "finger": finger,
                                    "xy_dist": xy_dist,
                                    "z_dist": z_dist,
                                    "confidence": tap.confidence,
                                },
                            )
                            print(
                                f"  tap: '{key}' "
                                f"({display_finger_name(finger)}, xy={xy_dist:.3f}, "
                                f"z={z_dist:.3f}, conf={tap.confidence:.2f})"
                            )
                            renderer.update_key_state(key, KeyState.PRESSED, 1.0, tap.confidence)

                    finger_states[finger] = _update_finger_state(finger_trails, finger, tx, ty, tz)

            frame = renderer.render(
                frame,
                finger_states=finger_states if config.visual.show_finger_trails else None,
                show_skeleton=config.visual.show_skeleton,
                hand_landmarks=[hf.raw_landmarks for hf in hand_frames] if config.visual.show_skeleton else None,
            )
            display_text = decoder.decoded_text[-100:]
            raw = decoder.get_raw_buffer()
            if raw:
                display_text += " " + raw
            cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.85, NAVY_BLUE, 2)
            if raw:
                cv2.putText(frame, f"Raw: {raw}", (20, 88), cv2.FONT_HERSHEY_DUPLEX, 0.5, (40, 30, 80), 1)
            cv2.putText(
                frame,
                f"Mode: {runtime_mode} | Decoder: {config.decoder.method}",
                (20, h - 20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.45,
                (120, 120, 130),
                1,
            )
            cv2.imshow("HoloType", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                decoder.decoded_text = ""
                decoder.buffer = []
                logger_session.write("cleared", {})
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

    duration = time.time() - start_time
    final_text = decoder.finalize()
    logger_session.write("typing_finished", {"text": final_text, "duration_seconds": duration})
    print("\n" + "=" * 60)
    print("  FINAL DECODED TEXT")
    print("=" * 60)
    print(final_text)
    if args.expected_text:
        metrics = score_text(args.expected_text, final_text, duration)
        logger_session.write("metrics", metrics_to_dict(metrics))
        print("\nValidation Metrics")
        print(f"  accuracy: {metrics.accuracy:.1%}")
        print(f"  CER:      {metrics.char_error_rate:.1%}")
        print(f"  WER:      {metrics.word_error_rate:.1%}")
        print(f"  WPM:      {metrics.wpm:.1f}")
    print("=" * 60)
    if logger_session.path:
        print(f"Session log: {logger_session.path}")
    return 0


def run_benchmark(config: AppConfig, args) -> int:
    """Score a saved session log or a direct expected/actual text pair."""
    if args.session:
        records = load_jsonl(args.session)
        expected = args.expected_text or ""
        actual = ""
        duration = 0.0
        for record in records:
            if record["type"] == "typing_started" and not expected:
                expected = record.get("expected_text", "")
            if record["type"] == "typing_finished":
                actual = record.get("text", "")
                duration = record.get("duration_seconds", 0.0)
        if not expected:
            print("No expected text found. Pass --expected-text for benchmarking.")
            return 1
    else:
        if not args.expected_text or args.actual_text is None:
            print("Benchmark needs --session, or both --expected-text and --actual-text.")
            return 1
        expected = args.expected_text
        actual = args.actual_text
        duration = args.duration

    metrics = score_text(expected, actual, duration)
    print("\nBenchmark Metrics")
    print(f"  expected: {metrics.expected}")
    print(f"  actual:   {metrics.actual}")
    print(f"  accuracy: {metrics.accuracy:.1%}")
    print(f"  CER:      {metrics.char_error_rate:.1%}")
    print(f"  WER:      {metrics.word_error_rate:.1%}")
    if duration:
        print(f"  WPM:      {metrics.wpm:.1f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HoloType - camera-only QWERTY typing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m src.main setup
  python3 -m src.main calibrate --mode single_finger
  python3 -m src.main type --expected-text "hello world"
  python3 -m src.main benchmark --session logs/sessions/latest.jsonl --expected-text "hello world"

Legacy flags still work:
  python3 -m src.main --calibrate --mode single_finger
  python3 -m src.main --type
        """,
    )
    parser.add_argument("command", nargs="?", choices=["setup", "calibrate", "type", "benchmark"])
    parser.add_argument("--calibrate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--type", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--mode", choices=["single_finger", "touch_typing"], help="Calibration mode")
    parser.add_argument("--smoothing", choices=["kalman", "one_euro", "none"], help="Smoothing method")
    parser.add_argument("--decoder", choices=["spellchecker", "noisy_channel"], help="Decoder method")
    parser.add_argument("--show-skeleton", action="store_true", help="Show hand skeleton overlay")
    parser.add_argument("--no-trails", action="store_true", help="Hide finger trails")
    parser.add_argument("--debug-tap", action="store_true", help="Enable tap detection debug output")
    parser.add_argument("--save-config", action="store_true", help="Persist CLI overrides to config file")
    parser.add_argument("--force-save", action="store_true", help="Save calibration even if quality checks fail")
    parser.add_argument("--log-session", action="store_true", default=True, help="Write JSONL session logs")
    parser.add_argument("--no-log-session", action="store_false", dest="log_session", help="Disable JSONL logs")
    parser.add_argument("--expected-text", default="", help="Target phrase for validation/benchmarking")
    parser.add_argument("--actual-text", default=None, help="Actual text for direct benchmark mode")
    parser.add_argument("--duration", type=float, default=0.0, help="Typing duration for direct benchmark mode")
    parser.add_argument("--session", default="", help="JSONL session log for benchmark mode")
    return parser


def load_config(path: str) -> AppConfig:
    if os.path.exists(path):
        return AppConfig.load(path)
    return DEFAULT_CONFIG


def apply_overrides(config: AppConfig, args) -> AppConfig:
    if args.mode:
        config.calibration.mode = args.mode
    if args.smoothing:
        config.smoothing.method = args.smoothing if args.smoothing != "none" else "kalman"
        config.smoothing.enabled = args.smoothing != "none"
    if args.decoder:
        config.decoder.method = args.decoder
    if args.show_skeleton:
        config.visual.show_skeleton = True
    if args.no_trails:
        config.visual.show_finger_trails = False
    return config


def resolve_command(args) -> str:
    if args.command:
        return args.command
    if args.calibrate:
        return "calibrate"
    if args.type:
        return "type"
    if args.benchmark:
        return "benchmark"
    return ""


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = resolve_command(args)
    config = apply_overrides(load_config(args.config), args)
    if args.save_config:
        config.save(args.config)

    if command == "setup":
        return run_setup(config, args)
    if command == "calibrate":
        return run_calibration(config, args)
    if command == "type":
        return run_typing(config, args)
    if command == "benchmark":
        return run_benchmark(config, args)

    parser.print_help()
    print("\nTry: python3 -m src.main setup")
    print("     python3 -m src.main calibrate --mode single_finger")
    print("     python3 -m src.main type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
