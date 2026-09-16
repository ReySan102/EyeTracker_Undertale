"""
Key injection: the only module that talks to the operating system's keyboard.

The game never sees "gaze"; it sees key-down and key-up events that look
exactly like a physical keyboard. Two kinds of key are needed:

  * held keys  - arrows during walking or dodging. The driver keeps a set of
                 what is currently down and, each tick, presses only what is
                 new and releases only what is gone (`apply`).
  * taps       - one-shot presses such as Z to confirm. A tap goes down now
                 and comes up a few frames later (`tap` + `update`), without
                 blocking the control loop.

Safety: `release_all()` runs on normal exit, on Ctrl+C and via atexit, so a
crash can never leave an arrow key stuck down in the middle of a boss fight.

Windows: DirectInputDriver uses pydirectinput, which sends hardware scan
codes through SendInput. GameMaker games accept those reliably.
Other platforms: see the notes in DESIGN.md (uinput on Linux, Quartz on macOS).

Run:  python -m eyetale.input_driver
"""
from __future__ import annotations

import atexit
import sys
from abc import ABC, abstractmethod
from typing import Iterable


class KeyDriver(ABC):
    """Base class: bookkeeping for held keys and pending tap releases."""

    def __init__(self) -> None:
        self.held: set[str] = set()
        self._taps: dict[str, float] = {}     # key -> time at which to release
        atexit.register(self.release_all)

    # -- platform hooks -----------------------------------------------------
    @abstractmethod
    def _key_down(self, key: str) -> None: ...

    @abstractmethod
    def _key_up(self, key: str) -> None: ...

    # -- public API ---------------------------------------------------------
    def apply(self, desired: Iterable[str]) -> None:
        """Make the set of held keys equal to `desired`, touching only the difference."""
        want = set(desired)

        for k in self.held - want:
            if k not in self._taps:
                self._key_up(k)

        for k in want - self.held:
            if k not in self._taps:
                self._key_down(k)

        self.held = want

    def tap(self, key: str, now: float, hold_s: float = 0.07) -> bool:
        """Press `key` now and schedule its release. Returns False if the key is already down (held or mid-tap), in which case nothing happens."""
        if key in self.held or key in self._taps:
            return False

        self._key_down(key)
        self._taps[key] = now + hold_s

        return True

    def update(self, now: float) -> None:
        """Release taps whose hold time has elapsed. Call once per tick."""
        for k, t_release in list(self._taps.items()):
            if now >= t_release:
                del self._taps[k]

                if k not in self.held:
                    self._key_up(k)

    def release_all(self) -> None:
        for k in list(self.held) + list(self._taps):
            try:
                self._key_up(k)
            except Exception:
                pass

        self.held.clear()
        self._taps.clear()

    @property
    def keys_down(self) -> set[str]:
        return self.held | set(self._taps)


class DryRunDriver(KeyDriver):
    """Prints and records events instead of sending them. Used by --dry-run and tests."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.events: list[tuple[str, str]] = []

    def _key_down(self, key: str) -> None:
        self.events.append(("down", key))

        if self.verbose:
            print(f"[dry-run] DOWN {key}")

    def _key_up(self, key: str) -> None:
        self.events.append(("up", key))

        if self.verbose:
            print(f"[dry-run] UP   {key}")


class DirectInputDriver(KeyDriver):
    """Real key injection on Windows via pydirectinput (scan codes + SendInput)."""

    def __init__(self) -> None:
        super().__init__()
        import pydirectinput  # Imported here so the rest of the project loads without it

        pydirectinput.PAUSE = 0.0          # The library sleeps 0.1 s per call by default
        pydirectinput.FAILSAFE = False
        self._pdi = pydirectinput

    def _key_down(self, key: str) -> None:
        self._pdi.keyDown(key)

    def _key_up(self, key: str) -> None:
        self._pdi.keyUp(key)


def make_driver(dry_run: bool = False, verbose: bool = True) -> KeyDriver:
    """Factory: real driver on Windows unless dry_run; DryRunDriver otherwise."""
    if dry_run or not sys.platform.startswith("win"):
        if not dry_run:
            print("[input_driver] not on Windows: using DryRunDriver")

        return DryRunDriver(verbose = verbose)

    try:
        return DirectInputDriver()
    except ImportError:
        print("[input_driver] pydirectinput not installed (pip install pydirectinput); using DryRunDriver")

        return DryRunDriver(verbose = verbose)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    d = DryRunDriver(verbose = False)

    # apply() presses only the difference.
    d.apply({"left", "up"})
    d.apply({"left", "up"})
    d.apply({"left"})
    assert d.events == [("down", "left"), ("down", "up"), ("up", "up")] or \
           d.events == [("down", "up"), ("down", "left"), ("up", "up")], d.events

    # A tap goes down now and up after hold_s.
    d.events.clear()
    assert d.tap("z", now = 0.0, hold_s = 0.07)
    assert not d.tap("z", now = 0.01)          # Already mid-tap
    d.update(0.05)
    assert d.events == [("down", "z")]
    d.update(0.08)
    assert d.events == [("down", "z"), ("up", "z")]

    # A tap on a key that is being held is a no-op; a held key that was tapped
    # first is not released when the tap expires.
    d.apply(set())
    d.events.clear()
    d.tap("up", now = 0.0, hold_s = 0.07)
    d.apply({"up"})                          # Already down: no second key-down
    d.update(1.0)                            # Tap expired but still held: no key-up
    assert d.events == [("down", "up")], d.events
    d.apply(set())
    assert d.events[-1] == ("up", "up")

    # release_all clears everything.
    d.apply({"left", "right"})
    d.tap("z", now = 0.0)
    d.release_all()
    assert not d.keys_down
    print("input_driver: OK")


if __name__ == "__main__":
    _self_test()
