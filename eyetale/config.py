"""
Configuration: every tunable number in the project lives here.

The config is a tree of dataclasses so that code gets autocomplete and type
checking, and it round-trips to a plain JSON file (config.json) so that you
can tune thresholds without touching code.

Run:  python -m eyetale.config          (self-test, also writes a default config.json
                                          if one does not exist yet)
"""
from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class KeysConfig:
    """Key names as pydirectinput understands them. Undertale defaults."""
    up: str = "up"
    down: str = "down"
    left: str = "left"
    right: str = "right"
    confirm: str = "z"
    cancel: str = "x"
    menu: str = "c"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Flip the preview so it behaves like a mirror. Processing always uses the raw frame.
    mirror_preview: bool = True


@dataclass
class FilterConfig:
    """
    One Euro filter parameters for the gaze point.

    min_cutoff: lower = smoother but laggier when the eye is still.
    beta:       higher = less lag during fast eye movements (saccades).
    """
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0


@dataclass
class GestureConfig:
    # Eye-closed thresholds are ratios of the open-eye EAR baseline measured
    # during calibration. Absolute fallbacks are used when no baseline exists.
    ear_closed_ratio: float = 0.60
    ear_open_ratio: float = 0.75
    ear_closed_abs: float = 0.18
    ear_open_abs: float = 0.22

    natural_blink_max_s: float = 0.25   # Shorter closures are ignored (normal blinks)
    long_blink_min_s: float = 0.35      # Deliberate blink -> confirm
    long_blink_max_s: float = 1.20      # Longer than this is not a "blink" any more
    double_blink_gap_s: float = 0.45    # Two short blinks within this gap -> cancel
    eyes_closed_pause_s: float = 1.60   # Eyes held shut this long -> toggle pause

    dwell_s: float = 0.60               # Stare at a HUD button this long to press it
    dwell_cooldown_s: float = 0.80      # Minimum time between two dwell presses

    gaze_hold_during_blink_s: float = 0.30  # Freeze the last command while eyes are shut
    face_lost_release_s: float = 0.50       # Face gone this long -> release every key


@dataclass
class OverworldConfig:
    """
    Compass control for walking and menus.

    Dead zone sizes are fractions of the game window width (x) and height (y). enter must be larger than exit: that gap is the hysteresis.
    """
    dead_zone_enter: float = 0.14
    dead_zone_exit: float = 0.09


@dataclass
class DodgeConfig:
    """Closed-loop SOUL control inside the bullet box. Pixels at 640x480; scaled automatically if the game window is bigger."""
    dead_zone_enter_px: float = 14.0
    dead_zone_exit_px: float = 6.0
    blue_jump_px: float = 28.0          # Look this far above the SOUL to jump (blue SOUL)
    green_aim_px: float = 40.0          # Look this far from box centre to aim the shield
    yellow_fire_interval_s: float = 0.25
    tap_hold_s: float = 0.07            # How long a one-shot key stays down (2 game frames)


@dataclass
class GameConfig:
    window_title: str = "UNDERTALE"
    capture_fps: int = 60
    # Used only on non-Windows systems or when the window cannot be found, so the
    # pipeline can still be exercised: left, top, width, height in screen pixels.
    fallback_rect: list[int] = field(default_factory = lambda: [100, 100, 640, 480])
    # Minimum SOUL pixels (at 640x480) before a colour blob counts as the SOUL.
    soul_min_pixels: int = 40
    # Colour tolerance per channel when matching the SOUL / UI colours.
    color_tolerance: int = 60


@dataclass
class HudConfig:
    enabled: bool = True
    width: int = 440
    height: int = 520
    gap_px: int = 12                    # Gap between the game window and the HUD
    show_camera: bool = True


@dataclass
class AppConfig:
    gaze_source: str = "webcam"         # Webcam | mouse | replay
    tick_hz: int = 60
    dry_run: bool = False               # Never send keys, only print them
    calibration_path: str = "calibration.json"
    model_path: str = "models/face_landmarker.task"
    log_path: str = "logs/gaze_log.csv"
    log_gaze: bool = False

    keys: KeysConfig = field(default_factory = KeysConfig)
    camera: CameraConfig = field(default_factory = CameraConfig)
    filter: FilterConfig = field(default_factory = FilterConfig)
    gestures: GestureConfig = field(default_factory = GestureConfig)
    overworld: OverworldConfig = field(default_factory = OverworldConfig)
    dodge: DodgeConfig = field(default_factory = DodgeConfig)
    game: GameConfig = field(default_factory = GameConfig)
    hud: HudConfig = field(default_factory = HudConfig)


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------

def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def _from_dict(cls: type, data: dict) -> Any:
    """Build a dataclass from a dict, recursing into nested dataclass fields and ignoring unknown keys (so old config files keep working)."""
    # `from __future__ import annotations` turns field types into strings, so
    # resolve them back into real classes before checking for nested dataclasses.
    hints = typing.get_type_hints(cls)
    kwargs = {}

    for f in fields(cls):
        if f.name not in data:
            continue

        value = data[f.name]
        ftype = hints.get(f.name, f.type)

        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)


def from_dict(data: dict) -> AppConfig:
    return _from_dict(AppConfig, data)


def default_config() -> AppConfig:
    return AppConfig()


def save_config(cfg: AppConfig, path: str | Path = "config.json") -> None:
    Path(path).write_text(json.dumps(to_dict(cfg), indent = 2), encoding = "utf-8")


def load_config(path: str | Path = "config.json") -> AppConfig:
    """Load config.json if it exists, otherwise return defaults."""
    p = Path(path)

    if not p.exists():
        return default_config()

    return from_dict(json.loads(p.read_text(encoding = "utf-8")))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import tempfile

    cfg = default_config()
    assert cfg.keys.confirm == "z"
    assert cfg.overworld.dead_zone_enter > cfg.overworld.dead_zone_exit
    assert cfg.dodge.dead_zone_enter_px > cfg.dodge.dead_zone_exit_px

    # Round trip through JSON preserves every value.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "config.json"
        cfg.gestures.dwell_s = 0.9
        save_config(cfg, p)
        back = load_config(p)
        assert back.gestures.dwell_s == 0.9
        assert to_dict(back) == to_dict(cfg)

        # Unknown keys and missing sections are tolerated.
        p.write_text(json.dumps({"tick_hz": 30, "bogus": 1, "keys": {"confirm": "enter", "nope": 2}}))
        partial = load_config(p)
        assert partial.tick_hz == 30 and partial.keys.confirm == "enter" and partial.keys.up == "up"

    if not Path("config.json").exists():
        save_config(default_config(), "config.json")
        print("wrote default config.json")

    print("config: OK")


if __name__ == "__main__":
    _self_test()
