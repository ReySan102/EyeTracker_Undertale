"""
Where is the game on the screen, and does it have keyboard focus?

WindowTracker polls the OS a couple of times per second (cheap) and caches:
  rect     client-area Rect of the Undertale window in screen pixels
  found    whether the window exists
  focused  whether it is the foreground window (keys are only sent when True)

On systems without win32 support (or when the window cannot be found and
`allow_fallback` is set) a fixed rectangle from the config is used so the
rest of the pipeline can be exercised in --dry-run mode.

Run:  python -m eyetale.game_window
"""
from __future__ import annotations

from .geometry import Rect
from . import winapi


class WindowTracker:
    def __init__(self, title: str, fallback_rect: list[int] | None = None, refresh_s: float = 0.5, allow_fallback: bool = False) -> None:
        self.title = title
        self.fallback = Rect(*fallback_rect) if fallback_rect else None
        self.refresh_s = refresh_s
        self.allow_fallback = allow_fallback or not winapi.IS_WINDOWS
        self.rect: Rect | None = None
        self.found = False
        self.focused = False
        self.using_fallback = False
        self._next_refresh = -1.0

    def update(self, now: float) -> None:
        if now < self._next_refresh:
            return

        self._next_refresh = now + self.refresh_s
        rect = winapi.find_window_rect(self.title)

        if rect is not None:
            self.rect, self.found, self.using_fallback = rect, True, False
            self.focused = winapi.foreground_title() == self.title
        elif self.allow_fallback and self.fallback is not None:
            self.rect, self.found, self.using_fallback = self.fallback, True, True
            self.focused = True
        else:
            self.rect, self.found, self.focused, self.using_fallback = None, False, False, False

    def focus(self) -> bool:
        return winapi.focus_window(self.title)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    wt = WindowTracker("no such window 987", fallback_rect = [10, 20, 640, 480], allow_fallback = True)
    wt.update(0.0)
    assert wt.found and wt.using_fallback and wt.rect == Rect(10, 20, 640, 480) and wt.focused

    strict = WindowTracker("no such window 987", fallback_rect = [10, 20, 640, 480], allow_fallback = False)
    strict.allow_fallback = False          # Force strict even off-Windows
    strict.update(0.0)
    assert not strict.found and strict.rect is None and not strict.focused

    # Refresh throttling: a second update inside refresh_s does not re-query.
    strict.rect = Rect(1, 1, 1, 1)
    strict.update(0.1)
    assert strict.rect == Rect(1, 1, 1, 1)
    strict.update(1.0)
    assert strict.rect is None
    print("game_window: OK")


if __name__ == "__main__":
    _self_test()
