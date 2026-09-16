"""
Entry point and the 60 Hz control loop.

    python -m eyetale.app gaze-demo     Step 1: See your gaze as a dot on a full-screen window
    python -m eyetale.app calibrate     Step 2: 13-point calibration -> calibration.json
    python -m eyetale.app test-keys     Step 3: Hold Right for one second in the game, tap Z
    python -m eyetale.app watch         Step 4: Show what the screen reader sees; sends no keys
    python -m eyetale.app play          Step 5: Play

Common options
    --source webcam|mouse|replay   Where gaze comes from (mouse = cursor, hold left button = eyes shut)
    --replay PATH                  CSV log for --source replay
    --dry-run                      Print key events instead of sending them
    --no-hud                       Do not open the HUD window
    --log                          Record gaze samples to logs/gaze_log.csv
    --config PATH                  Config file (default config.json, created if missing)

The loop, once per tick:
    window tracker -> gaze sample -> screen reader -> gestures -> HUD dwell -> mapper -> key driver -> HUD render -> sleep
"""
from __future__ import annotations

import argparse
import sys
import time

from pathlib import Path
from . import winapi
from .calibration import CalibrationModel, run_calibration
from .config import AppConfig, load_config, save_config
from .game_window import WindowTracker
from .gaze_estimator import GazeLogger, GazeSource, WebcamGazeSource, make_gaze_source
from .gestures import BlinkDetector, DwellDetector
from .hud import Hud
from .input_driver import KeyDriver, make_driver
from .mapper import Mapper, Mode, WindowInfo
from .screen_reader import ScreenReader, StateMemory, classify_state, draw_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_source(cfg: AppConfig, replay: str | None) -> GazeSource:
    model = CalibrationModel.load(cfg.calibration_path) if cfg.gaze_source == "webcam" else None

    if cfg.gaze_source == "webcam" and model is None:
        print("[app] no calibration.json yet: gaze will be invalid until you run `calibrate`")

    src = make_gaze_source(cfg, model, replay)

    if not src.start():
        print("[app] gaze source failed to start")
        sys.exit(2)

    return src


def _calibrate_with(src: GazeSource, cfg: AppConfig) -> CalibrationModel | None:
    if not isinstance(src, WebcamGazeSource):
        print("[app] calibration needs the webcam source")

        return None

    w, h = winapi.screen_size()
    model = run_calibration(src.latest_features, w, h)

    if model is None:
        print("[app] calibration aborted")

        return None

    model.save(cfg.calibration_path)
    src.set_model(model)
    print(f"[app] saved {cfg.calibration_path}: RMSE {model.rmse_px:.0f} px, EAR baseline {model.ear_baseline:.2f}")

    return model


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_calibrate(cfg: AppConfig, args) -> None:
    cfg.gaze_source = "webcam"
    src = _start_source(cfg, None)

    try:
        time.sleep(0.5)
        _calibrate_with(src, cfg)
    finally:
        src.stop()


def cmd_gaze_demo(cfg: AppConfig, args) -> None:
    """Full-screen window with the filtered gaze (magenta), the raw gaze (grey) and blink events. Esc to quit."""
    import cv2
    import numpy as np

    src = _start_source(cfg, args.replay)
    w, h = winapi.screen_size()
    name = "EyeTale gaze demo"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas = np.zeros((h, w, 3), np.uint8)
    blink = BlinkDetector(cfg.gestures, src.ear_baseline)
    last_event, last_event_t = "", -1e9

    try:
        while True:
            now = time.perf_counter()
            g = src.latest()
            canvas[:] = (16, 16, 16)

            for x in range(0, w, w // 8):
                cv2.line(canvas, (x, 0), (x, h), (40, 40, 40), 1)

            for y in range(0, h, h // 6):
                cv2.line(canvas, (0, y), (w, y), (40, 40, 40), 1)

            msg = "no gaze sample"

            if g is not None:
                if g.valid:
                    for ev in blink.update(g.ear, now):
                        last_event, last_event_t = ev.name, now

                    cv2.circle(canvas, (int(g.raw_x), int(g.raw_y)), 6, (90, 90, 90), 1)
                    cv2.circle(canvas, (int(g.x), int(g.y)), 14, (255, 0, 255), 2)
                    msg = f"gaze ({g.x:.0f}, {g.y:.0f})  EAR {g.ear:.2f}  {'CLOSED' if blink.closed else 'open'}"
                else:
                    f = src.latest_features()
                    msg = ("face found but no calibration: run `python -m eyetale.app calibrate`"
                           if f is not None and f.ok else "no face detected")

            if now - last_event_t < 1.0:
                cv2.putText(canvas, last_event, (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 220, 120), 3)

            cv2.putText(canvas, msg + "   (Esc quits)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            if isinstance(src, WebcamGazeSource):
                cv2.putText(canvas, f"tracker {src.fps:.0f} fps", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)

            cv2.imshow(name, canvas)

            if cv2.waitKey(16) & 0xFF == 27:
                break
    finally:
        cv2.destroyAllWindows()
        src.stop()


def cmd_test_keys(cfg: AppConfig, args) -> None:
    """Prove key injection reaches the game before building anything on top."""
    driver = make_driver(args.dry_run)
    tracker = WindowTracker(cfg.game.window_title, cfg.game.fallback_rect, allow_fallback = args.dry_run)
    tracker.update(time.perf_counter())

    if not tracker.found:
        print(f"[test-keys] window '{cfg.game.window_title}' not found: start the game first (windowed)")

        if not args.dry_run:
            return

    tracker.focus()
    print("[test-keys] switch to the game. Holding Right for 1 s, then tapping Z, in:")

    for i in (3, 2, 1):
        print(f"  {i}")
        time.sleep(1.0)

    try:
        driver.apply({cfg.keys.right})
        time.sleep(1.0)
        driver.apply(set())
        time.sleep(0.3)
        t0 = time.perf_counter()
        driver.tap(cfg.keys.confirm, t0, cfg.dodge.tap_hold_s)

        while time.perf_counter() - t0 < 0.3:
            driver.update(time.perf_counter())
            time.sleep(0.01)
    finally:
        driver.release_all()

    print("[test-keys] done. Did the character walk right and did Z register?")


def cmd_play(cfg: AppConfig, args, send_keys: bool = True) -> None:
    src = _start_source(cfg, args.replay)
    driver: KeyDriver = make_driver(dry_run = args.dry_run or not send_keys, verbose = args.dry_run)
    tracker = WindowTracker(cfg.game.window_title, cfg.game.fallback_rect, allow_fallback = args.dry_run or not send_keys)

    try:
        reader: ScreenReader | None = ScreenReader()
    except Exception as e:  # pragma: no cover
        print(f"[app] screen capture unavailable ({e}); running without game-state detection")
        reader = None

    blink = BlinkDetector(cfg.gestures, src.ear_baseline)
    dwell = DwellDetector(cfg.gestures)
    mapper = Mapper(cfg)
    memory = StateMemory()
    hud = Hud(cfg) if (cfg.hud.enabled and not args.no_hud) else None
    logger = GazeLogger(cfg.log_path) if (args.log or cfg.log_gaze) else None

    if hud:
        hud.open()

    period = 1.0 / cfg.tick_hz
    capture_period = 1.0 / max(1, cfg.game.capture_fps)
    last_capture = -1e9
    frame = state = None
    placed_for = None
    tick = 0
    loop_fps, n_ticks, t_fps = 0.0, 0, time.perf_counter()
    last_mode: Mode | None = None
    print("[app] running. Esc in the HUD window quits; Ctrl+C in this terminal also works.")

    try:
        while True:
            now = time.perf_counter()
            tracker.update(now)
            rect = tracker.rect

            # Park the HUD beside the game once we know where the game is.
            if hud and rect is not None and placed_for != rect:
                hud.place_next_to(rect)
                placed_for = rect
                tracker.focus()

            if hud:
                hud.update_rect(now)

            gaze = src.latest()

            # Game state from the screen, at capture_fps.
            if rect is not None and reader is not None and now - last_capture >= capture_period:
                try:
                    frame = reader.grab(rect)
                    state = memory.update(classify_state(frame, cfg.game, memory.fallback_box(now)), now)
                except Exception as e:  # Window moved off-screen, etc.
                    frame, state = None, None

                    if tick % 300 == 0:
                        print(f"[app] capture failed: {e}")

                last_capture = now
            elif rect is None:
                frame = state = None

            # Gestures.
            events = blink.update(gaze.ear, now) if (gaze is not None and gaze.valid) else []
            hud_target = hud.hit_test(gaze.x, gaze.y) if (hud and gaze is not None and gaze.valid) else None
            hud_action = dwell.update(hud_target, now)

            if hud_action == "recalibrate":
                driver.release_all()
                model = _calibrate_with(src, cfg)

                if model is not None:
                    blink = BlinkDetector(cfg.gestures, model.ear_baseline)

                hud_action = None

            # Decide and act.
            cmd = mapper.update(gaze, WindowInfo(rect, tracker.focused), state, events, hud_action, blink.closed, now)
            driver.apply(cmd.held)

            for key in cmd.taps:
                driver.tap(key, now, cfg.dodge.tap_hold_s)

            driver.update(now)

            if cmd.mode != last_mode:
                print(f"[app] mode -> {cmd.mode.name} {cmd.info}")
                last_mode = cmd.mode

            # HUD at half rate.
            if hud and tick % 2 == 0:
                gaze_local = None
                hud_local = None

                if gaze is not None and gaze.valid:
                    if rect is not None:
                        gaze_local = rect.to_local(gaze.x, gaze.y)

                    if hud.rect is not None:
                        hud_local = hud.rect.to_local(gaze.x, gaze.y)

                view = None

                if frame is not None and state is not None:
                    view = draw_state(frame, state, gaze_local, cfg.dodge.dead_zone_enter_px * state.scale)

                cam = None

                if cfg.hud.show_camera:
                    cam_frame = src.latest_frame()

                    if cam_frame is not None:
                        from .face_tracker import draw_features

                        f = src.latest_features()
                        cam = draw_features(cam_frame, f) if f is not None else cam_frame

                        if cfg.camera.mirror_preview:
                            import cv2

                            cam = cv2.flip(cam, 1)

                src_fps = getattr(src, "fps", 0.0)

                lines = [
                    f"mode: {cmd.mode.name.lower()}   held: {' '.join(sorted(cmd.held)) or '-'}",
                    f"{cmd.info or (state.mode_name if state else '')}",
                    f"gaze: {gaze.x:.0f},{gaze.y:.0f}  ear {gaze.ear:.2f}" if gaze else "gaze: none",
                    f"fps  gaze {src_fps:.0f}   loop {loop_fps:.0f}   {'DRY RUN' if args.dry_run or not send_keys else 'keys live'}",
                    f"soul: {state.soul_color} {tuple(int(v) for v in state.soul)}" if state and state.soul else "soul: -",
                    f"window: {rect.as_tuple() if rect else 'not found'}{' (fallback)' if tracker.using_fallback else ''}",
                ]

                hud.render(lines, view, cam, hud_target, dwell.progress(now), hud_local)

                if hud.poll_key() == 27:
                    break

            if logger:
                logger.log(gaze)

            # Loop rate.
            tick += 1
            n_ticks += 1

            if now - t_fps >= 1.0:
                loop_fps, n_ticks, t_fps = n_ticks / (now - t_fps), 0, now

            if getattr(src, "finished", False):        # Replay reached the end of the log
                break

            elapsed = time.perf_counter() - now

            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        driver.release_all()
        src.stop()

        if hud:
            hud.close()

        if reader:
            reader.close()

        if logger:
            logger.close()

        print("[app] stopped, all keys released")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog = "python -m eyetale.app", description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices = ["gaze-demo", "calibrate", "test-keys", "watch", "play"])
    p.add_argument("--config", default = "config.json")
    p.add_argument("--source", choices = ["webcam", "mouse", "replay"])
    p.add_argument("--replay", help = "CSV log to replay (with --source replay)")
    p.add_argument("--dry-run", action = "store_true")
    p.add_argument("--no-hud", action = "store_true")
    p.add_argument("--log", action = "store_true")

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    winapi.set_dpi_aware()               # Before any window is created
    cfg = load_config(args.config)

    if not Path(args.config).exists():
        save_config(cfg, args.config)
        print(f"[app] wrote default {args.config}")

    if args.source:
        cfg.gaze_source = args.source

    if args.replay:
        cfg.gaze_source = "replay"

    if args.command == "calibrate":
        cmd_calibrate(cfg, args)
    elif args.command == "gaze-demo":
        cmd_gaze_demo(cfg, args)
    elif args.command == "test-keys":
        cmd_test_keys(cfg, args)
    elif args.command == "watch":
        if not args.source:
            cfg.gaze_source = "mouse"

        cmd_play(cfg, args, send_keys = False)
    elif args.command == "play":
        cmd_play(cfg, args, send_keys = True)


if __name__ == "__main__":
    main()
