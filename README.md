# HoloType

Camera-only QWERTY typing: type on an imaginary keyboard in front of your webcam. No physical or on-screen keyboard required.

- MediaPipe hand tracking for one-hand
- Single-finger and touch-typing calibration with handed finger IDs
- Z-velocity tap detection with smoothing
- Noisy-channel or spellchecker decoding
- Guided setup, calibration quality checks, session logs, and benchmark metrics
- OpenCV keyboard overlay with tap feedback, trails, and optional skeleton debug view

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 -m src.main setup
python3 -m src.main calibrate --mode single_finger
python3 -m src.main type
```

After installing the package, the same commands are available through:

```bash
holo-type setup
holo-type calibrate --mode single_finger
holo-type type
```

## Recommended Workflow

1. Run `setup` to confirm the camera and hand tracking are usable.
2. Start with `calibrate --mode single_finger`; it is faster and easier to tune.
3. Type a known phrase with validation enabled:

```bash
python3 -m src.main type --expected-text "hello world"
```
4. Check the printed accuracy, CER, WER, and WPM.


## Commands

| Command     | Purpose                                                   |
| ----------- | --------------------------------------------------------- |
| `setup`     | Camera and hand-tracking sanity check                     |
| `calibrate` | Guided calibration; saves only when the profile is usable |
| `type`      | Live air typing with decoder and session logging          |
| `benchmark` | Score expected vs actual text or a saved session          |


## Useful Options

```bash
python3 -m src.main calibrate --mode touch_typing
python3 -m src.main type --show-skeleton
python3 -m src.main type --smoothing one_euro
python3 -m src.main type --decoder spellchecker
python3 -m src.main type --no-log-session
python3 -m src.main calibrate --force-save
```

CLI overrides are temporary by default. Add `--save-config` when you want to persist them to `config.json`.

## Controls

| Key | Action                                  |
| --- | --------------------------------------- |
| `q` | Quit                                    |
| `c` | Clear typed text                        |
| `r` | Redo current calibration key            |
| `s` | Save calibration if quality checks pass |

## Tap Gesture

Push the active finger forward/down toward the camera with a deliberate tap motion. The detector looks for:

1. Fingertip moving below the PIP joint
2. Negative Z velocity spike
3. Positive Z rebound

## Requirements

- Python 3.10+
- Webcam, ideally 720p+ at 30 FPS
- Good lighting

## TODO

- Improve word typing accuracy: words are not always identified correctly, and the decoding logic may need further tuning.
