"""
Discrete gestures from continuous signals.

Gaze gives you a position, but the game also needs "press" events. Those come
from what the eyes do over time:

  BlinkDetector      eye aspect ratio over time -> LONG_BLINK (confirm),
                     DOUBLE_BLINK (cancel), EYES_CLOSED_HOLD (pause toggle)
  DwellDetector      "the gaze has rested on target X for N ms" -> fire X once
  FaceLostWatchdog   "the face has been gone for N ms" -> release every key

All three are pure state machines driven by (value, timestamp) so they can be
tested with synthetic sequences and replayed logs.

Run:  python -m eyetale.gestures
"""
from __future__ import annotations

from enum import Enum, auto

from .config import GestureConfig


class GestureEvent(Enum):
    LONG_BLINK = auto()         # Deliberate blink, ~0.35-1.2 s  -> confirm (Z)
    DOUBLE_BLINK = auto()       # Two quick blinks              -> cancel (X)
    EYES_CLOSED_HOLD = auto()   # Eyes shut > 1.6 s              -> pause / resume


class BlinkDetector:
    """
    Classifies eye closures by duration.

    Hysteresis on the EAR threshold (closed below `closed_thr`, open again only
    above `open_thr`) stops a half-closed eye from generating a burst of
    micro-blinks.
    """

    def __init__(self, cfg: GestureConfig, ear_baseline: float = 0.0) -> None:
        self.cfg = cfg

        if ear_baseline > 0.05:
            self.closed_thr = ear_baseline * cfg.ear_closed_ratio
            self.open_thr = ear_baseline * cfg.ear_open_ratio
        else:
            self.closed_thr = cfg.ear_closed_abs
            self.open_thr = cfg.ear_open_abs

        self.closed = False
        self.closed_since = 0.0
        self._hold_fired = False
        self._last_short_blink: float | None = None
        self.last_event: GestureEvent | None = None
        self.last_event_t = -1e9

    def closed_for(self, t: float) -> float:
        return (t - self.closed_since) if self.closed else 0.0

    def update(self, ear: float, t: float) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        c = self.cfg

        if not self.closed:
            if ear < self.closed_thr:
                self.closed, self.closed_since, self._hold_fired = True, t, False

            return events

        # Eyes are closed
        if ear > self.open_thr:
            self.closed = False
            dur = t - self.closed_since

            if self._hold_fired:
                pass                                        # Already consumed as a hold
            elif dur <= c.natural_blink_max_s:
                if self._last_short_blink is not None and t - self._last_short_blink <= c.double_blink_gap_s:
                    events.append(GestureEvent.DOUBLE_BLINK)
                    self._last_short_blink = None
                else:
                    self._last_short_blink = t
            elif c.long_blink_min_s <= dur <= c.long_blink_max_s:
                events.append(GestureEvent.LONG_BLINK)
                self._last_short_blink = None
            # Durations in the gap between natural and long, or beyond long_max, are ignored
        elif not self._hold_fired and t - self.closed_since >= c.eyes_closed_pause_s:
            self._hold_fired = True
            events.append(GestureEvent.EYES_CLOSED_HOLD)

        if events:
            self.last_event, self.last_event_t = events[-1], t

        return events


class DwellDetector:
    """Fires once when the same target has been looked at for `dwell_s`."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self.target: str | None = None
        self.since = 0.0
        self._fired = False
        self._last_fire = -1e9

    def progress(self, t: float) -> float:
        if self.target is None or self._fired:
            return 0.0

        return max(0.0, min(1.0, (t - self.since) / self.cfg.dwell_s))

    def update(self, target: str | None, t: float) -> str | None:
        if target != self.target:
            self.target, self.since, self._fired = target, t, False

            return None

        if target is None or self._fired:
            return None

        if t - self.since >= self.cfg.dwell_s and t - self._last_fire >= self.cfg.dwell_cooldown_s:
            self._fired = True
            self._last_fire = t

            return target

        return None


class FaceLostWatchdog:
    """True once the gaze has been invalid for longer than `face_lost_release_s`."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._lost_since: float | None = None

    def update(self, valid: bool, t: float) -> bool:
        if valid:
            self._lost_since = None

            return False

        if self._lost_since is None:
            self._lost_since = t

        return t - self._lost_since >= self.cfg.face_lost_release_s


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run(det: BlinkDetector, pattern: list[tuple[float, float]]) -> list[GestureEvent]:
    """pattern: (ear, duration_s) segments sampled at 60 Hz."""
    out, t = [], 0.0

    for ear, dur in pattern:
        n = max(1, int(dur * 60))

        for _ in range(n):
            out += det.update(ear, t)
            t += 1 / 60

    return out


def _self_test() -> None:
    cfg = GestureConfig()
    OPEN, SHUT = 0.30, 0.05

    # Natural blink (0.15 s): no event.
    d = BlinkDetector(cfg, ear_baseline = OPEN)
    assert _run(d, [(OPEN, 1), (SHUT, 0.15), (OPEN, 1)]) == []

    # Long blink (0.5 s): LONG_BLINK.
    d = BlinkDetector(cfg, ear_baseline = OPEN)
    assert _run(d, [(OPEN, 1), (SHUT, 0.5), (OPEN, 1)]) == [GestureEvent.LONG_BLINK]

    # Two quick blinks 0.2 s apart: DOUBLE_BLINK, exactly once.
    d = BlinkDetector(cfg, ear_baseline = OPEN)
    ev = _run(d, [(OPEN, 1), (SHUT, 0.12), (OPEN, 0.2), (SHUT, 0.12), (OPEN, 1)])
    assert ev == [GestureEvent.DOUBLE_BLINK], ev

    # Two natural blinks 2 s apart: nothing.
    d = BlinkDetector(cfg, ear_baseline = OPEN)
    assert _run(d, [(OPEN, 1), (SHUT, 0.12), (OPEN, 2.0), (SHUT, 0.12), (OPEN, 1)]) == []

    # Eyes held shut 2 s: EYES_CLOSED_HOLD fires while closed, and reopening adds nothing.
    d = BlinkDetector(cfg, ear_baseline = OPEN)
    ev = _run(d, [(OPEN, 1), (SHUT, 2.0), (OPEN, 1)])
    assert ev == [GestureEvent.EYES_CLOSED_HOLD], ev

    # Hysteresis: EAR hovering between the two thresholds does not toggle.
    d = BlinkDetector(cfg, ear_baseline = OPEN)
    mid = (d.closed_thr + d.open_thr) / 2
    assert _run(d, [(OPEN, 0.5), (SHUT, 0.05), (mid, 0.5), (OPEN, 0.5)]) == [GestureEvent.LONG_BLINK]

    # Absolute fallback thresholds when no baseline is known.
    d = BlinkDetector(cfg, ear_baseline = 0.0)
    assert d.closed_thr == cfg.ear_closed_abs

    # Dwell: fires once after dwell_s, not again until the target changes, respects cooldown.
    dw = DwellDetector(cfg)
    fired = [dw.update("confirm", t / 60) for t in range(120)]
    assert fired.count("confirm") == 1 and fired.index("confirm") >= int(cfg.dwell_s * 60) - 1
    assert dw.update(None, 2.0) is None
    assert dw.update("confirm", 2.01) is None and dw.progress(2.3) > 0
    assert dw.update("confirm", 2.01 + cfg.dwell_s) == "confirm"

    # Watchdog.
    wd = FaceLostWatchdog(cfg)
    assert not wd.update(False, 0.0) and not wd.update(False, 0.3) and wd.update(False, 0.6)
    assert not wd.update(True, 0.7) and not wd.update(False, 0.8)
    print("gestures: OK")


if __name__ == "__main__":
    _self_test()
