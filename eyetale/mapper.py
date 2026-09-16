"""
The mapper: gaze + gestures + game state  ->  keys.

This is the brain of the project and it is deliberately pure: it never
touches the camera, the screen or the keyboard. Every tick it receives what
the other modules measured and returns a `Command`:

    held   the set of keys that should be down right now (arrows)
    taps   one-shot presses to fire this tick (Z / X / C)
    mode   which control scheme produced them (for the HUD)

Control schemes
  OVERWORLD / BATTLE_MENU   compass: the game window is split into a centre
                            dead zone and four directions. Look left of the
                            dead zone -> Left is held until you look back.
                            Per-axis Schmitt triggers give hysteresis, and the
                            two axes are independent so diagonals work.
  DODGE_RED/YELLOW/PURPLE   closed loop: the SOUL chases the gaze. Error =
                            gaze - SOUL position, per axis, through the same
                            trigger with a pixel dead zone.
  DODGE_BLUE                like red on x; on y a larger threshold above the
                            SOUL holds Up (jump; longer hold = higher jump).
  DODGE_GREEN               the SOUL does not move; the dominant direction of
                            gaze from the box centre taps an arrow to turn the
                            shield, once per change of direction.

Global rules
  * A long blink taps confirm, a double blink taps cancel, HUD dwell buttons
    tap whatever they are labelled with.
  * Eyes-closed-hold (or the HUD pause button) toggles PAUSED: nothing is held.
  * While the eyes are shut mid-blink, the previous held keys are kept for a
    short grace period so a blink does not stop the SOUL mid-dodge.
  * No game window, game not focused, or face lost -> release everything.
  * Gaze far outside the game window (e.g. on the HUD) -> release everything.

Run:  python -m eyetale.mapper
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .config import AppConfig
from .filters import SchmittTrigger
from .gaze_estimator import GazeSample
from .geometry import Rect
from .gestures import FaceLostWatchdog, GestureEvent
from .screen_reader import GameState


class Mode(Enum):
    PAUSED = auto()
    NO_GAME = auto()
    UNFOCUSED = auto()
    NO_FACE = auto()
    OVERWORLD = auto()
    BATTLE_MENU = auto()
    DODGE_RED = auto()
    DODGE_BLUE = auto()
    DODGE_GREEN = auto()
    DODGE_YELLOW = auto()
    DODGE_PURPLE = auto()


DODGE_MODES = {"red": Mode.DODGE_RED, "blue": Mode.DODGE_BLUE, "green": Mode.DODGE_GREEN,
               "yellow": Mode.DODGE_YELLOW, "purple": Mode.DODGE_PURPLE}

HUD_TAP_ACTIONS = ("confirm", "cancel", "menu")


@dataclass
class Command:
    held: set[str] = field(default_factory = set)
    taps: list[str] = field(default_factory = list)
    mode: Mode = Mode.NO_GAME
    info: str = ""


@dataclass
class WindowInfo:
    """The subset of WindowTracker the mapper needs (keeps the mapper testable)."""
    rect: Rect | None
    focused: bool


class Mapper:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.k = cfg.keys
        self.paused = False
        self._ow_x = SchmittTrigger(1.0, 0.5)
        self._ow_y = SchmittTrigger(1.0, 0.5)
        self._dg_x = SchmittTrigger(cfg.dodge.dead_zone_enter_px, cfg.dodge.dead_zone_exit_px)
        self._dg_y = SchmittTrigger(cfg.dodge.dead_zone_enter_px, cfg.dodge.dead_zone_exit_px)
        self._watchdog = FaceLostWatchdog(cfg.gestures)
        self._mode = Mode.NO_GAME
        self._last_held: set[str] = set()
        self._last_fire_t = -1e9
        self._green_aim: str | None = None
        self._eyes_closed_since: float | None = None

    # -- helpers ------------------------------------------------------------
    def _reset_triggers(self) -> None:
        for t in (self._ow_x, self._ow_y, self._dg_x, self._dg_y):
            t.reset()

        self._green_aim = None

    def _set_mode(self, mode: Mode) -> None:
        if mode != self._mode:
            self._mode = mode
            self._reset_triggers()

    def _axes_to_keys(self, sx: int, sy: int) -> set[str]:
        held = set()

        if sx < 0:
            held.add(self.k.left)
        elif sx > 0:
            held.add(self.k.right)

        if sy < 0:
            held.add(self.k.up)
        elif sy > 0:
            held.add(self.k.down)

        return held

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._reset_triggers()

    # -- main entry ---------------------------------------------------------
    def update(self, gaze: GazeSample | None, window: WindowInfo, state: GameState | None,
               events: list[GestureEvent], hud_action: str | None, eyes_closed: bool, now: float) -> Command:
        taps: list[str] = []

        if GestureEvent.EYES_CLOSED_HOLD in events or hud_action == "pause":
            self.toggle_pause()

        if self.paused:
            return self._finish(Command(set(), [], Mode.PAUSED, "paused: hold eyes shut to resume"))

        if window.rect is None:
            self._set_mode(Mode.NO_GAME)

            return self._finish(Command(set(), [], Mode.NO_GAME, "game window not found"))

        if not window.focused:
            self._set_mode(Mode.UNFOCUSED)

            return self._finish(Command(set(), [], Mode.UNFOCUSED, "click on the game window"))

        valid = gaze is not None and gaze.valid

        if self._watchdog.update(valid, now):
            self._set_mode(Mode.NO_FACE)

            return self._finish(Command(set(), [], Mode.NO_FACE, "face not found"))

        if not valid:
            return Command(set(self._last_held), [], self._mode, "gaze dropout: holding last command")

        # Discrete actions from gestures and HUD dwell buttons.
        if GestureEvent.LONG_BLINK in events:
            taps.append(self.k.confirm)

        if GestureEvent.DOUBLE_BLINK in events:
            taps.append(self.k.cancel)

        if hud_action in HUD_TAP_ACTIONS:
            taps.append(getattr(self.k, hud_action))

        # Mid-blink: the gaze estimate is garbage, keep the previous command briefly.
        if eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now

            if now - self._eyes_closed_since <= self.cfg.gestures.gaze_hold_during_blink_s:
                return Command(set(self._last_held), taps, self._mode, "blink: holding")

            return self._finish(Command(set(), taps, self._mode, "eyes closed"))

        self._eyes_closed_since = None

        rect = window.rect
        gx, gy = rect.to_local(gaze.x, gaze.y)

        # Looking well outside the game (HUD, second monitor): stand still.
        mx, my = rect.width * 0.25, rect.height * 0.25

        if not (-mx <= gx < rect.width + mx and -my <= gy < rect.height + my):
            self._reset_triggers()

            return self._finish(Command(set(), taps, self._mode, "gaze outside game"))

        # Pick the control scheme.
        if state is not None and state.in_battle and state.soul_in_box:
            mode = DODGE_MODES.get(state.soul_color or "red", Mode.DODGE_RED)
        elif state is not None and state.in_battle:
            mode = Mode.BATTLE_MENU
        else:
            mode = Mode.OVERWORLD

        self._set_mode(mode)

        if mode in (Mode.OVERWORLD, Mode.BATTLE_MENU):
            held = self._compass(gx, gy, rect)
        elif mode == Mode.DODGE_GREEN:
            held, tap = self._green(gx, gy, state)

            if tap:
                taps.append(tap)
        else:
            held = self._dodge(gx, gy, state, mode)

            if mode == Mode.DODGE_YELLOW and now - self._last_fire_t >= self.cfg.dodge.yellow_fire_interval_s:
                taps.append(self.k.confirm)
                self._last_fire_t = now

        return self._finish(Command(held, taps, mode, ""))

    def _finish(self, cmd: Command) -> Command:
        self._last_held = set(cmd.held)

        return cmd

    # -- schemes ------------------------------------------------------------
    def _compass(self, gx: float, gy: float, rect: Rect) -> set[str]:
        ow = self.cfg.overworld
        self._ow_x.set_thresholds(ow.dead_zone_enter * rect.width, ow.dead_zone_exit * rect.width)
        self._ow_y.set_thresholds(ow.dead_zone_enter * rect.height, ow.dead_zone_exit * rect.height)
        sx = self._ow_x.update(gx - rect.width / 2.0)
        sy = self._ow_y.update(gy - rect.height / 2.0)

        return self._axes_to_keys(sx, sy)

    def _dodge(self, gx: float, gy: float, state: GameState, mode: Mode) -> set[str]:
        dg = self.cfg.dodge
        s = state.scale
        box, (soul_x, soul_y) = state.box, state.soul
        # Looking outside the box still steers toward its edge, never beyond it.
        gx, gy = box.clamp_point(gx, gy)
        self._dg_x.set_thresholds(dg.dead_zone_enter_px * s, dg.dead_zone_exit_px * s)

        if mode == Mode.DODGE_BLUE:
            self._dg_y.set_thresholds(dg.blue_jump_px * s, dg.blue_jump_px * s * 0.5)
        else:
            self._dg_y.set_thresholds(dg.dead_zone_enter_px * s, dg.dead_zone_exit_px * s)

        sx = self._dg_x.update(gx - soul_x)
        sy = self._dg_y.update(gy - soul_y)

        return self._axes_to_keys(sx, sy)

    def _green(self, gx: float, gy: float, state: GameState) -> tuple[set[str], str | None]:
        cx, cy = state.box.center
        dx, dy = gx - cx, gy - cy
        thr = self.cfg.dodge.green_aim_px * state.scale

        if max(abs(dx), abs(dy)) < thr:
            return set(), None

        if abs(dx) >= abs(dy):
            aim = self.k.left if dx < 0 else self.k.right
        else:
            aim = self.k.up if dy < 0 else self.k.down

        if aim == self._green_aim:
            return set(), None

        self._green_aim = aim

        return set(), aim


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    from .config import default_config

    cfg = default_config()
    m = Mapper(cfg)
    rect = Rect(100, 100, 640, 480)
    win = WindowInfo(rect, True)
    ow = GameState(in_battle = False, frame_w = 640, frame_h = 480)
    k = cfg.keys

    def g(x, y, valid = True, ear = 0.3):
        return GazeSample(0.0, x + rect.left, y + rect.top, valid, ear)

    def tick(gz, state = ow, events = (), hud = None, closed = False, now = 0.0, w = win):
        return m.update(gz, w, state, list(events), hud, closed, now)

    # Overworld compass with hysteresis.
    c = tick(g(320, 240))
    assert c.mode == Mode.OVERWORLD and c.held == set(), c
    assert tick(g(320 + 0.20 * 640, 240)).held == {k.right}          # Past enter (14 %)
    assert tick(g(320 + 0.11 * 640, 240)).held == {k.right}          # Inside band (9-14 %): keep
    assert tick(g(320 + 0.05 * 640, 240)).held == set()              # Below exit: release
    assert tick(g(320 - 0.20 * 640, 240 + 0.20 * 480)).held == {k.left, k.down}
    assert tick(g(320, 240)).held == set()

    # Gestures and HUD taps.
    assert tick(g(320, 240), events = [GestureEvent.LONG_BLINK]).taps == [k.confirm]
    assert tick(g(320, 240), events = [GestureEvent.DOUBLE_BLINK]).taps == [k.cancel]
    assert tick(g(320, 240), hud = "menu").taps == [k.menu]

    # Blink grace: keys are kept briefly while the eyes are shut, then released.
    tick(g(320 + 0.2 * 640, 240), now = 1.0)
    assert tick(g(320, 240, ear = 0.0), closed = True, now = 1.1).held == {k.right}
    assert tick(g(320, 240, ear = 0.0), closed = True, now = 1.5).held == set()
    tick(g(320, 240), now = 1.6)

    # Pause toggle.
    assert tick(g(320, 240), events = [GestureEvent.EYES_CLOSED_HOLD]).mode == Mode.PAUSED
    assert tick(g(320 + 0.3 * 640, 240)).held == set()
    assert tick(g(320, 240), hud = "pause").mode == Mode.OVERWORLD

    # Focus and window presence.
    assert tick(g(320 + 0.3 * 640, 240), w = WindowInfo(rect, False)).mode == Mode.UNFOCUSED
    assert tick(g(320 + 0.3 * 640, 240), w = WindowInfo(None, True)).mode == Mode.NO_GAME

    # Face lost: short dropout holds the last command, long dropout releases.
    tick(g(320 + 0.2 * 640, 240), now = 5.0)
    assert tick(g(0, 0, valid = False), now = 5.1).held == {k.right}
    assert tick(g(0, 0, valid = False), now = 6.0).mode == Mode.NO_FACE
    tick(g(320, 240), now = 6.1)

    # Gaze far outside the window releases everything.
    tick(g(320 + 0.2 * 640, 240))
    assert tick(g(640 + 400, 240)).held == set()

    # Battle menu uses the compass too.
    menu = GameState(in_battle = True, box = Rect(37, 255, 566, 140), frame_w = 640, frame_h = 480)
    assert tick(g(320 - 0.2 * 640, 240), state = menu).mode == Mode.BATTLE_MENU
    assert tick(g(320 - 0.2 * 640, 240), state = menu).held == {k.left}

    # Dodge (red): the SOUL chases the gaze with a pixel dead zone.
    box = Rect(200, 250, 240, 150)
    red = GameState(in_battle = True, box = box, soul = (320.0, 330.0), soul_color = "red", frame_w = 640, frame_h = 480)
    assert tick(g(360, 330), state = red).mode == Mode.DODGE_RED
    assert tick(g(360, 330), state = red).held == {k.right}
    assert tick(g(328, 330), state = red).held == {k.right}            # 8 px: inside band, keep
    assert tick(g(324, 330), state = red).held == set()                # 4 px: released
    assert tick(g(300, 300), state = red).held == {k.left, k.up}
    assert tick(g(320, 470), state = red).held == {k.down}             # Below the box: clamped to its edge

    # Blue: looking well above the SOUL jumps.
    blue = GameState(in_battle = True, box = box, soul = (320.0, 380.0), soul_color = "blue", frame_w = 640, frame_h = 480)
    assert tick(g(320, 380 - 10), state = blue).held == set()
    assert tick(g(320, 380 - 40), state = blue).held == {k.up}

    # Green: one tap per change of aim direction.
    green = GameState(in_battle = True, box = box, soul = (320.0, 325.0), soul_color = "green", frame_w = 640, frame_h = 480)
    c1 = tick(g(200, 325), state = green)
    assert c1.mode == Mode.DODGE_GREEN and c1.taps == [k.left] and c1.held == set()
    assert tick(g(205, 325), state = green).taps == []
    assert tick(g(320, 250), state = green).taps == [k.up]

    # Yellow: auto-fire at the configured interval.
    yellow = GameState(in_battle = True, box = box, soul = (320.0, 330.0), soul_color = "yellow", frame_w = 640, frame_h = 480)
    assert tick(g(320, 330), state = yellow, now = 10.0).taps == [k.confirm]
    assert tick(g(320, 330), state = yellow, now = 10.1).taps == []
    assert tick(g(320, 330), state = yellow, now = 10.3).taps == [k.confirm]
    print("mapper: OK")


if __name__ == "__main__":
    _self_test()
