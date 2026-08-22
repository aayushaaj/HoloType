# Air Typer

Camera-only QWERTY typing — type on an imaginary keyboard in front of your webcam. No physical or on-screen keyboard required.

## Features

- **Hand tracking** via MediaPipe Hands (21 landmarks, ~30 FPS)
- **Multi-finger calibration** — single-finger or full 10-finger touch-typing
- **Z-velocity tap detection** with adaptive thresholds
- **Noisy-channel decoder** — spatial confidence + n-gram LM + word frequency
- **Glassmorphism UI** — iOS-style frosted glass keyboard at 60 FPS
- **Kalman / One-Euro smoothing** for stable fingertip positions

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Calibrate (first run)
python -m src.main --calibrate --mode single_finger   # ~1 min, index finger only
# or
python -m src.main --calibrate --mode touch_typing    # ~3 min, all 10 fingers

# Type
python -m src.main --type
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear text |
| `r` | Redo current key (calibration only) |

## Tap Gesture

Push your index finger **forward/down toward the camera** — a deliberate air-tap motion. The detector looks for:
1. Finger tip moving below PIP joint (curled press)
2. Negative Z velocity spike
3. Positive Z rebound

## Project Structure

```
air-typer/
├── src/
│   ├── hand_tracker.py    # MediaPipe Hands wrapper
│   ├── smoothing.py       # Kalman & One-Euro filters
│   ├── tap_detector.py    # Z-velocity tap detection
│   ├── calibration.py     # Multi-finger calibration
│   ├── decoder.py         # Noisy-channel + spellchecker
│   ├── visuals.py         # Glassmorphism renderer
│   ├── config.py          # Typed configuration
│   └── main.py            # CLI orchestration
├── data/
│   ├── calibration_profile.json
│   └── models/            # MediaPipe model (downloaded on demand)
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Options

```bash
# Debug tap detection
python -m src.main --type --debug-tap

# Show hand skeleton
python -m src.main --type --show-skeleton

# Use One-Euro smoothing
python -m src.main --type --smoothing one_euro
```

## Requirements

- Python 3.10+
- Webcam (720p+ @ 30fps recommended)
- Good lighting (MediaPipe needs visible hand)

## License

MIT