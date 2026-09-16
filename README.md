# EyeTale

Control Undertale using nothing but your eyes. A webcam feeds MediaPipe's face tracker, and a mapping layer turns your gaze and blinks into the game's own keyboard presses — the game itself stays untouched.

See `DESIGN.md` for the deep dive: every module, every function, the order things get built in, and how the tuning works.

## Quick start (Windows)

```
python -m venv .venv
.venv\Scripts\activate      # ***
pip install -r requirements.txt
python run_tests.py

python -m eyetale.app gaze-demo      # camera + face tracking working?
python -m eyetale.app calibrate      # 13-point calibration -> calibration.json
python -m eyetale.app test-keys      # holds Right for 1 s in the game, taps Z
python -m eyetale.app watch          # shows what the screen reader sees, sends no keys
python -m eyetale.app play           # play
```

Play with Undertale windowed rather than fullscreened via F4. Handy flags along the way: `--dry-run`, `--source mouse`, `--log`, `--source replay --replay logs/gaze_log.csv`, `--no-hud`.

*** If encountered error: "cannot be loaded because running scripts is disabled on this system" run:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\activate

With RemoteSigned, scripts you wrote yourself still run, but anything downloaded still has to carry a signature — which is exactly why Microsoft points people toward it for cases like this. Prefer not to touch a permanent setting? `Set-ExecutionPolicy -Scope Process Bypass` gets you the same result for just the current window, though you'll need to run it again each time you open a new terminal.

## Layout

```
eyetale/
  config.py          every tunable, config.json round trip
  filters.py         One Euro filter, Schmitt trigger
  geometry.py        Rect and coordinate conversion
  input_driver.py    key injection (pydirectinput) with held-key diffing and taps
  winapi.py          window / cursor / DPI helpers with fallbacks
  camera.py          webcam thread
  face_tracker.py    MediaPipe Face Landmarker -> iris offsets, EAR, head pose
  calibration.py     13-point calibration, ridge regression, calibration.json
  gaze_estimator.py  webcam / mouse / replay gaze sources, logging
  gestures.py        long blink, double blink, eyes-closed hold, dwell
  game_window.py     find the Undertale window, focus check
  screen_reader.py   battle / bullet box / SOUL detection from pixels
  mapper.py          gaze + gestures + game state -> keys (the state machine)
  hud.py             side window: feedback and dwell buttons
  app.py             commands and the 60 Hz loop
run_tests.py         runs every module's self-test
```
