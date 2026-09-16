# EyeTale: playing Undertale with your eyes

EyeTale is a Python program that turns a webcam into an Undertale controller. Your gaze moves the character and the SOUL, a deliberate blink confirms, a double blink cancels, and a row of dwell buttons next to the game handles the menu key and pausing. The game itself is never touched: EyeTale watches the pixels of the game window to figure out what's happening, then sends ordinary keyboard events, so as far as Undertale is concerned a person is pressing the arrow keys and Z.

This document walks through how the whole thing fits together, what every file and function is for, and the order the pieces were built in, so you can either rebuild it from scratch or extend it.

## The idea in one paragraph

Undertale only ever needs a handful of keys: the four arrows, Z (confirm), X (cancel) and C (menu). Every moment in the game boils down to one of four situations: walking around, choosing in a menu, dodging inside the bullet box, or timing the FIGHT bar. So the program's whole job is answering, sixty times a second, "which keys should be down right now, and should any key get tapped this instant?" Answering that needs three inputs: where the eyes are pointing (gaze), what the eyes are doing over time (blinks, dwelling), and what the game is showing (battle or not, where the box is, where the SOUL is). Each of those gets its own module. A state machine combines them into a key set, and a driver makes that key set true on the keyboard.

## How one tick works

The control loop in `app.py` runs at 60 Hz. Within a single tick:

1. `WindowTracker` reports where the Undertale window sits on screen and whether it currently has keyboard focus.
2. The gaze source hands over its newest `GazeSample`: a screen coordinate, a validity flag, and the eye aspect ratio (how open the eyes are).
3. `ScreenReader` grabs the game's client area; `classify_state` turns those pixels into a `GameState` (in battle? where's the box? where's the SOUL and what color?). `StateMemory` carries the last good box and SOUL forward for a fraction of a second so a brief flicker doesn't knock the program out of dodge mode.
4. `BlinkDetector` watches the eye aspect ratio and reports gestures: a long blink, a double blink, or eyes held shut. `DwellDetector` checks whether the gaze has rested on a HUD button long enough to trigger it.
5. `Mapper.update` takes everything above and returns a `Command`: the keys to hold, the keys to tap, and which mode produced them.
6. `KeyDriver.apply` presses and releases only the difference between what's currently held and what should be held; `tap` presses a one-shot key and schedules its release two game frames later.
7. Every other tick the HUD redraws: the game view with the gaze cursor, box and SOUL overlaid, the status lines, and the dwell buttons with their progress bars.
8. The loop sleeps through whatever remains of the 16.7 ms budget.

Steps 4 through 6 are pure Python with no hardware dependency at all, which is exactly why each of those modules ships with a self-test that runs anywhere, and why a recorded gaze log can be replayed through the entire pipeline with `--source replay`.

## Setup

Windows 10/11 is the target platform. The game needs to run windowed (F4 toggles fullscreen in Undertale — leave it off) so the HUD can sit next to it and the screen reader can locate it. The default 640x480 window works fine; a larger integer scale works too, since pixel thresholds scale with the window width.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_tests.py
```

The first time anything touches the camera, MediaPipe's `face_landmarker.task` model (about 3.7 MB) gets downloaded into `models/`. A `config.json` populated with every default is written the first time the app runs — tune by editing that rather than the code.

## The commands, in the order you use them

`python -m eyetale.app gaze-demo` opens a full-screen window showing where the program thinks you're looking. Before calibration it can only report whether a face is found and the eye aspect ratio, which is enough to confirm the camera and tracker are working. Pass `--source mouse` to watch the same screen driven by the mouse instead.

`python -m eyetale.app calibrate` shows thirteen dots, one at a time. Keep your head still, track each dot with your eyes, and the program fits a model mapping your eye features to screen pixels. It also measures your open-eye baseline, which sets the blink thresholds. The results screen reports the mean error at each dot; anything under roughly 80 px is comfortable for this control scheme.

`python -m eyetale.app test-keys` counts down, holds Right for one second, and taps Z — proving synthesized input reaches the game before anything gets built on top of it. Add `--dry-run` to print the events instead.

`python -m eyetale.app watch` runs the full loop without sending any keys, with the mouse standing in for gaze by default. Play a battle with the keyboard and watch the HUD: the bullet box should get a green outline, the SOUL a cyan circle, and the mode line should read `dodge:red` during attacks and `battle-menu` between them.

`python -m eyetale.app play` is the real thing. `--dry-run` prints keys rather than sending them, `--source mouse` lets the cursor drive things (hold the left button to "close your eyes"), `--log` records the gaze stream to `logs/gaze_log.csv`, and `--source replay --replay logs/gaze_log.csv` plays a log like that back through the pipeline.

## Module by module

The modules are listed in dependency order — each one only imports what's above it.

### config.py

Every tunable number lives inside a tree of dataclasses (`KeysConfig`, `CameraConfig`, `FilterConfig`, `GestureConfig`, `OverworldConfig`, `DodgeConfig`, `GameConfig`, `HudConfig`, all nested under `AppConfig`). `save_config` and `load_config` round-trip that tree through `config.json`; `from_dict` walks the dataclass fields recursively and skips unknown keys, so an older config file keeps working even after new fields get added. Since the file uses `from __future__ import annotations`, field types exist as strings at runtime, and `typing.get_type_hints` resolves them back into real classes before the nested check runs. The self-test confirms both the round trip and the tolerance for partial files.

### filters.py

Two pure signal tools live here. `OneEuroFilter` is an adaptive low-pass filter: it smooths aggressively when the input barely moves and loosens up as the input speeds up, so gaze jitter disappears without adding lag to saccades. `min_cutoff` controls smoothing at rest, and `beta` controls how fast the filter loosens as speed increases. `OneEuroFilter2D` just runs one of these per axis.

`SchmittTrigger` turns a signed error value into -1, 0, or +1 with hysteresis built in: the state flips to +1 only once the value clears `enter`, and only returns to 0 once it drops below the smaller `exit`. That gap between the two thresholds is exactly what stops an arrow key from flickering when gaze sits right on a zone boundary. The mapper runs four of these, one per axis per control scheme.

### geometry.py

`Rect` is the one shape used everywhere — the game window, the bullet box, HUD buttons. It converts between screen coordinates (absolute desktop pixels, where gaze and windows live) and window-local coordinates (relative to the game's top-left corner, where the screen reader and mapper operate). `clamp_point` keeps a gaze that's wandered outside the bullet box pinned to its edge.

### input_driver.py

`KeyDriver` is the bookkeeping base class. `apply(desired)` compares the desired key set against `self.held` and only presses what's new while releasing what's gone, so a key held for five seconds produces exactly one key-down event. `tap(key, now, hold_s)` presses a key and records when it should release; `update(now)` releases any taps that are due. The interplay between the two is handled deliberately: tapping a key that's already held is a no-op, and a held key that started as a tap doesn't get released when that tap expires. `release_all` is registered with `atexit` and also called from the loop's `finally` block, so a crash can never leave an arrow key stuck down.

`DirectInputDriver` is the Windows implementation, built on pydirectinput, which sends hardware scan codes through `SendInput`; GameMaker games accept these in cases where plain virtual-key events sometimes get ignored. Its per-call pause is set to zero because the library otherwise sleeps 100 ms per call by default, which would wreck the loop's timing. `DryRunDriver` just prints and records events; `make_driver` picks whichever is appropriate.

### winapi.py

Every operating-system call lives in this one file, each with a fallback so the project still imports on any OS. `set_dpi_aware` gets called before any window is created; skip it and Windows reports scaled coordinates to Python on 125% displays, and nothing lines up anymore. `screen_size`, `cursor_pos`, and `mouse_button_down` come from user32 via ctypes. `find_window_rect` uses pywin32 to fetch a window's client area by title and converts it into screen coordinates; `foreground_title` reports which window currently has focus; `focus_window` and `move_window` do exactly what their names say.

### camera.py

`CameraThread` reads frames on a background thread, since `VideoCapture.read` blocks for a full frame period. It keeps only the newest frame along with a capture timestamp and a sequence number, so a slow consumer drops frames rather than falling behind. On Windows it opens the camera through DirectShow and sets the buffer size to one — both choices cut down latency.

### face_tracker.py

`FaceTracker` wraps MediaPipe's Face Landmarker in VIDEO mode with blendshapes and the facial transformation matrix turned on. `process(frame, t, seq)` returns a `FaceFeatures` object. The interesting logic lives in the pure helpers:

`iris_offset` expresses the iris center relative to the midpoint between the eye corners, divided by the eye width. That makes the resulting number independent of where the face sits in the frame, of camera resolution, and of eyelid position — which is exactly what lets a regression map it onto screen position.

`eye_aspect_ratio` is the classic six-point EAR: the sum of two vertical eyelid distances divided by twice the horizontal eye width. Open eyes land around 0.25 to 0.35, closed ones under 0.15. `FaceFeatures.ear` takes the minimum across both eyes, so a wink still counts as closed.

`head_pose_from_matrix` extracts yaw, pitch, roll, and translation from MediaPipe's 4x4 matrix. `feature_vector` packs the iris offsets, their squares and cross terms, the head pose, and a handful of iris-times-head interaction terms into a 21-value vector. The squares let a linear fit bend slightly, and the head terms let it compensate for small head movements.

`features_from_landmarks` builds all of this from a (478, 2) pixel array — which is also what the self-test feeds it, using synthetic eye geometry. `ensure_model` handles downloading the model file on first use.

### calibration.py

`calibration_points` returns thirteen screen fractions: a 3x3 grid plus four inner points. `run_calibration` shows each dot, waits a second for the eye to settle, gathers roughly 1.2 s of feature vectors (skipping any frames mid-blink), and then calls `fit_model`.

`fit_model` standardizes every feature column and solves ridge regression in closed form, `W = (ZᵀZ + λI)⁻¹ ZᵀY`, leaving the bias column unpenalized. Ridge regression rather than plain least squares, because 21 correlated features fitted from just thirteen points would otherwise overfit badly. The result is a `CalibrationModel` exposing `predict(vec) -> (x, y)`, the open-eye EAR baseline, the RMSE, and the per-point errors; `save`/`load` round-trip it through `calibration.json`. `evaluate_per_point` generates the numbers shown on the results screen, so a bad corner is obvious right away.

### gaze_estimator.py

`GazeSample` is the unit everything downstream trades in: time, filtered x/y, validity, EAR, raw x/y, and a sequence number. `GazeSource` is the interface; the control loop only ever calls `latest()`, which never blocks.

`WebcamGazeSource` runs on its own thread: newest camera frame, `FaceTracker.process`, `CalibrationModel.predict`, `OneEuroFilter2D`, then store the sample. With no model yet it still produces features (so calibration and the gaze demo both work) but flags the samples invalid.

`MouseGazeSource` is the development stand-in: the cursor acts as gaze, and holding the left mouse button reports an EAR of zero. This is what makes it possible to build and test everything downstream of the tracker before the tracker even works, and to demo the mapping to someone without calibrating them first.

`GazeLogger` writes samples out to CSV; `ReplayGazeSource` reads them back either in real time or one sample per call, for deterministic tests. `make_gaze_source` is the factory the app relies on.

### gestures.py

`BlinkDetector.update(ear, t)` is a small state machine. The eye counts as "closed" once EAR drops below `closed_thr`, and "open" again only once it clears the higher `open_thr`; both are ratios of the calibration baseline. On reopening, the closure duration decides what event fires: 0.25 s or under is a natural blink and gets ignored (unless a second one follows within 0.45 s, which makes a DOUBLE_BLINK); 0.35 to 1.2 s counts as a LONG_BLINK; and if the eyes are still closed past 1.6 s, an EYES_CLOSED_HOLD fires once. The self-test exercises it with synthetic EAR sequences covering each case.

`DwellDetector.update(target, t)` fires a target once the gaze has rested on it for `dwell_s`, then stays quiet until the target changes; `progress` feeds the HUD's fill bars. `FaceLostWatchdog` returns True once the gaze has been invalid for half a second, which the mapper treats as a signal to release everything.

### game_window.py

`WindowTracker` polls `winapi.find_window_rect` twice a second and caches the rectangle along with a `focused` flag. Keys only get sent while the game is the foreground window — the single most important safety rule in the whole project: alt-tab to a browser, and EyeTale stops pressing arrows immediately. Off Windows, or in dry-run mode, a fallback rectangle from the config steps in so the loop can still run.

### screen_reader.py

This is how the program figures out what the game is doing. `ScreenReader.grab` uses mss to capture the client area as a BGR array. The detection functions themselves are pure:

`detect_battle_ui` scans the bottom 16% of the window for orange or yellow contours whose bounding boxes match the size and shape of the FIGHT/ACT/ITEM/MERCY buttons; three or more of those means a battle is on. Matching shape rather than just counting orange pixels is what keeps Hotland's orange floors from being mistaken for a battle.

`find_bullet_box` thresholds near-white pixels, pulls the external contours, and keeps the largest one that covers at least 1.2% of the window, forms a proper rectangle (contour area close to its bounding box area), and has a mostly dark interior. The white enemy sprite fails the rectangle test; the overworld dialogue box never even reaches this code since it's only called during battle. What comes back is the interior rectangle, with the 5 px border stripped off.

`find_soul` masks for each SOUL color (red, blue, green, yellow, purple) inside the box and returns the centroid of whichever blob is biggest above a minimum pixel count. Searching only inside the box is intentional: the HP bar is red and yellow, and the SOUL sits on the orange buttons during the menu phase, so anything outside the box would throw the detection off. The color name doubles as the SOUL mode.

`classify_state` strings all of this together into a `GameState`, and `StateMemory` layers in short-term memory: the SOUL blinks during invincibility frames, and a dense attack can hide the border for a frame, so the last good box gets reused for up to 0.5 s and the last SOUL position for 0.25 s. `draw_state` renders the annotated view for the HUD, and `synthetic_battle_frame` builds a fake battle screen for the self-test, complete with a white enemy blob, an HP bar, and decoy text inside the box.

### mapper.py

`Mapper.update` is the decision-maker, and it's pure: it takes the gaze sample, a `WindowInfo` (rect and focus), the `GameState`, the gesture events, the HUD action, and the current time, and returns a `Command`. The order these checks run in matters:

1. An EYES_CLOSED_HOLD or the HUD pause button toggles `paused`; while paused, nothing is held.
2. No window or no focus: hold nothing.
3. Gaze invalid: hold the last command for up to half a second (the watchdog), then release.
4. Gestures turn into taps: long blink -> confirm, double blink -> cancel, HUD buttons -> their assigned key.
5. Eyes shut mid-blink: keep the previously held keys for 0.3 s so a blink doesn't interrupt a dodge, then release.
6. Gaze well outside the game window (25% margin): hold nothing. This is what lets you look at the HUD without also walking.
7. Pick a scheme based on game state and run it.

The compass scheme (`_compass`) covers the overworld and battle menus. Error is gaze minus window center, per axis, run through a Schmitt trigger whose thresholds are fractions of the window size (enter at 14%, exit at 9%). The two axes are independent, so looking to the lower-left holds Left and Down together. Since menus in Undertale only react to fresh key presses, moving two entries left means glancing back to center and then left again — which turns out to feel natural in practice.

The dodge scheme (`_dodge`) is closed-loop: error is gaze minus SOUL position in window pixels, clamped to the bullet box, run through triggers with a pixel dead zone (enter 14 px, exit 6 px, scaled by window size). The SOUL moves at a fixed speed in the game, so an on/off controller with hysteresis is the right tool for the job — it walks the SOUL toward wherever you're looking and stops there. Blue SOUL uses the same horizontal control but a larger vertical threshold (28 px), so that looking clearly above the SOUL holds Up, which jumps, and holding it longer jumps higher. Yellow adds an automatic confirm tap every 250 ms to keep shooting. Purple behaves the same as red.

The green scheme (`_green`) doesn't move the SOUL at all — it picks the dominant axis of gaze relative to the box center beyond a 40 px threshold and taps that arrow once per direction change, which rotates the shield.

Switching modes resets every trigger, so no stale state carries over from one scheme into the next. The self-test exercises every branch with synthetic samples: hysteresis bands, diagonals, blink grace, pause, focus loss, face loss, the outside-window rule, and each SOUL color.

### hud.py

`Hud` is an OpenCV window parked next to the game (`place_next_to`). It renders status lines, a camera thumbnail with the eye landmarks, the annotated game view, and five dwell buttons: Z, X, C, PAUSE, and CAL. Hit-testing (`hit_test`) needs the buttons' screen rectangles, which come from `winapi.find_window_rect` on the HUD's own title, so the buttons keep working no matter where the window ends up. The fill bar on the active button shows dwell progress; Esc with the HUD focused quits.

### app.py

`build_parser` defines the five commands and their options. `cmd_calibrate`, `cmd_gaze_demo`, and `cmd_test_keys` are the step-by-step tools described above. `cmd_play` wires the loop together exactly as described in "How one tick works"; `watch` is the same loop with keys disabled and the mouse as the default source. `_calibrate_with` runs calibration against a live webcam source and swaps in the new model — which is also what the HUD's CAL button triggers mid-session. The `finally` block releases every key, stops the source, closes the HUD and the screen reader, and closes the log.

### run_tests.py

Imports each module in build order and runs its `_self_test()`. Hardware-dependent tests report SKIP instead of failing.

## Build order from scratch

This is the order the project was actually written in, and the order to follow if you're rebuilding it. Each step leaves you with something runnable and checkable before you move to the next. Functions are listed in the order they were written within each step.

1. `config.py`: `KeysConfig` through `AppConfig`, then `to_dict`, `_from_dict`, `from_dict`, `default_config`, `save_config`, `load_config`. At this point you have a config file and a home for every number that follows.
2. `filters.py`: `LowPassFilter.apply`, `OneEuroFilter.filter`, `OneEuroFilter2D.filter`, `SchmittTrigger.update`. Both algorithms are testable against synthetic numbers before any hardware exists.
3. `geometry.py`: `Rect` with `contains`, `to_local`, `to_screen`, `shrink`, `clamp_point`; `distance`.
4. `input_driver.py`: `KeyDriver.apply`, `KeyDriver.tap`, `KeyDriver.update`, `KeyDriver.release_all`, then `DryRunDriver`, `DirectInputDriver`, `make_driver`. Check point: `python -m eyetale.app test-keys` (written later, but this is where the driver actually gets proven against the real game — do it now with a five-line script if you'd like).
5. `winapi.py`: `set_dpi_aware`, `screen_size`, `cursor_pos`, `mouse_button_down`, `find_window_rect`, `foreground_title`, `focus_window`, `move_window`.
6. `camera.py`: `CameraThread.start`, `_loop`, `latest`, `stop`. Check point: the self-test prints the frame size and measured fps.
7. `face_tracker.py`: pure helpers first (`eye_aspect_ratio`, `iris_offset`, `head_pose_from_matrix`, `feature_vector`, `features_from_landmarks`), then `ensure_model`, `FaceTracker.process`, `draw_features`.
8. `calibration.py`: `calibration_points`, `fit_model`, `CalibrationModel.predict`/`save`/`load`, `evaluate_per_point`, then the on-screen `run_calibration`.
9. `gaze_estimator.py`: `GazeSample`, `GazeSource`, `MouseGazeSource` (so everything downstream can be developed with the mouse first), `WebcamGazeSource`, `GazeLogger`, `ReplayGazeSource`, `make_gaze_source`. Check point: `gaze-demo`, then `calibrate`, then `gaze-demo` again with a real gaze dot.
10. `gestures.py`: `BlinkDetector.update`, `DwellDetector.update`/`progress`, `FaceLostWatchdog.update`. Check point: the gaze demo prints LONG_BLINK / DOUBLE_BLINK when you perform them.
11. `game_window.py`: `WindowTracker.update`, `focus`.
12. `screen_reader.py`: `color_mask`, `detect_battle_ui`, `find_bullet_box`, `find_soul`, `classify_state`, `StateMemory.update`, `ScreenReader.grab`, `draw_state`, and the synthetic frame builders for the test. Check point: `watch` while playing a battle with the keyboard.
13. `mapper.py`: `Mode`, `Command`, `WindowInfo`, `Mapper._compass`, `_dodge`, `_green`, `_axes_to_keys`, `update`. This step carries the most tests — write them alongside the code.
14. `hud.py`: `Hud.open`, `place_next_to`, `update_rect`, `hit_test`, `render`, `poll_key`, `close`.
15. `app.py`: `_start_source`, `_calibrate_with`, `cmd_calibrate`, `cmd_gaze_demo`, `cmd_test_keys`, `cmd_play`, `build_parser`, `main`.
16. `run_tests.py`.

## What to do with your eyes

| Situation | Eyes | Keys sent |
|---|---|---|
| Walk | Look left / right / up / down of the window centre; look back at the centre to stop | Arrow held |
| Talk, advance text, confirm | Close your eyes for about half a second | Z tap |
| Cancel, back out of a menu | Two quick blinks | X tap |
| Open the menu | Look at the HUD's C button until it fills | C tap |
| Choose FIGHT / ACT / ITEM / MERCY | Glance left or right, back to centre, repeat; long blink to pick | Arrow, then Z |
| FIGHT timing bar | Long blink when the bar is near the centre (blink a touch early: latency is about 0.1 s) | Z tap |
| Dodge (red SOUL) | Look where you want the SOUL to be; it follows | Arrows held |
| Blue SOUL | Look above the SOUL to jump, longer for higher | Up held |
| Green SOUL | Look toward the incoming arrow's side to turn the shield | Arrow tap |
| Yellow SOUL | Aim as red; shooting is automatic | Z taps |
| Pause / resume | Close your eyes for about two seconds, or dwell on PAUSE | All released |
| Recalibrate | Dwell on CAL | Runs calibration |

Cyan bullets (stay still) mean "look at the SOUL"; orange bullets (keep moving) behave the way you'd expect.

## Tuning

Most problems trace back to a single number in `config.json`.

The character twitches, or a key flickers at the zone edge: widen the hysteresis gap by lowering `overworld.dead_zone_exit` or raising `dead_zone_enter`. The SOUL oscillates around the gaze point: raise `dodge.dead_zone_enter_px`. Movement lags behind your eyes: raise `filter.beta` a little, or raise `filter.min_cutoff`. Gaze feels jittery: lower `filter.min_cutoff` (0.6 is a reasonable floor).

Natural blinks are triggering confirm: raise `gestures.long_blink_min_s`. Long blinks aren't registering: lower that same value, or check the EAR baseline on the calibration results screen and adjust `ear_closed_ratio`. Double blinks fire accidentally: lower `double_blink_gap_s`. Confirm fires while you're reading text: that's expected behavior, since reading is exactly when deliberate blinks tend to happen — use the HUD's Z button instead whenever reading matters.

Battles aren't getting detected: run `watch`, check the bottom band in the HUD, and loosen `game.color_tolerance` or the size ranges inside `detect_battle_ui`. The SOUL gets lost mid-attack: raise `game.color_tolerance` slightly and double-check the color table in `screen_reader.py` matches your build of the game. The box gets lost during dense attacks: raise the interior-darkness limit inside `find_bullet_box`.

Calibration RMSE above roughly 100 px: add more light on the face, put the camera at eye level, sit 50 to 70 cm away, keep your head still during calibration, and recalibrate if you shift in your chair.

## Limits and where to go next

Latency from a 30 fps webcam runs roughly 80 to 120 ms end to end (frame period, inference, filter, the game's own 30 Hz polling). That's enough for the pacifist route and most bosses, but not enough for the game's hardest attacks. A 60 fps camera would help; a dedicated tracker would help more. Tobii's consumer Eye Tracker 5 is officially gaming-only, and its developer SDK is paid and restricted to other models, so going that route means expecting to rely on a community bridge — the `GazeSource` interface is exactly where that would plug in.

Webcam accuracy (roughly 2 to 5 degrees, so 50 to 150 px) is why the control schemes are built around zones and dead zones instead of direct pointing. Head movement after calibration is the biggest source of drift; the head-pose features soften that but don't eliminate it, which is the whole reason the CAL button exists.

Other platforms: the key driver would need a uinput virtual keyboard on Linux (python-evdev) and `CGEventPost` on macOS; `winapi.py` would need equivalent window queries written for each. Everything else is portable as-is.

A translucent overlay drawn directly over the game (gaze cursor, zone outline) would look nicer than the side HUD; it would need a layered, click-through window instead of OpenCV, which makes it a self-contained follow-up project.
