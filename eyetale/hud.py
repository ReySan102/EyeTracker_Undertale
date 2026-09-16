"""
The HUD: a small OpenCV window parked beside the game.

It does three jobs:
  * feedback  - shows the gaze point over a live copy of the game window,
                the detected bullet box and SOUL, the current mode, held keys
                and frame rates, so you can see why the program did what it did
  * buttons   - a row of dwell buttons (confirm / cancel / menu / pause /
                recalibrate). Stare at one for `dwell_s` and it fires. Their
                screen rectangles come from the HUD window's own position, so
                hit-testing works wherever the window is.
  * escape    - press Esc with the HUD focused to quit cleanly

Undertale has no mouse support and OpenCV windows are opaque, so the HUD sits
next to the game instead of over it. Run the game windowed for this to work.

Run:  python -m eyetale.hud       (renders a demo frame; press any key)
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .config import AppConfig
from .geometry import Rect
from . import winapi

BG = (28, 26, 24)
PANEL = (44, 42, 40)
TEXT = (220, 220, 220)
DIM = (140, 140, 140)
ACCENT = (255, 200, 0)
FIRE = (80, 220, 120)

BUTTONS: list[tuple[str, str]] = [
    ("confirm", "Z"),
    ("cancel", "X"),
    ("menu", "C"),
    ("pause", "PAUSE"),
    ("recalibrate", "CAL"),
]


class Hud:
    NAME = "EyeTale HUD"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.w, self.h = cfg.hud.width, cfg.hud.height
        self.rect: Rect | None = None               # Client rect in screen px
        self._next_rect_refresh = -1.0
        self._canvas = np.zeros((self.h, self.w, 3), np.uint8)
        # Local button rectangles along the bottom edge.
        n = len(BUTTONS)
        pad, bh = 8, 64
        bw = (self.w - pad * (n + 1)) // n
        self.buttons: list[tuple[str, str, Rect]] = [
            (bid, label, Rect(pad + i * (bw + pad), self.h - bh - pad, bw, bh)) for i, (bid, label) in enumerate(BUTTONS)
        ]
        self._open = False

    # -- window management --------------------------------------------------
    def open(self) -> None:
        if cv2 is None:
            return

        cv2.namedWindow(self.NAME, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.NAME, self._canvas)
        cv2.waitKey(1)
        self._open = True

    def place_next_to(self, game: Rect) -> None:
        if not self._open:
            return

        cv2.moveWindow(self.NAME, game.right + self.cfg.hud.gap_px, game.top)
        cv2.waitKey(1)
        self._next_rect_refresh = -1.0

    def update_rect(self, now: float) -> None:
        if now < self._next_rect_refresh:
            return

        self._next_rect_refresh = now + 0.5
        self.rect = winapi.find_window_rect(self.NAME)

    def hit_test(self, x: float, y: float) -> str | None:
        """Which dwell button (if any) is under screen point (x, y)?"""
        if self.rect is None:
            return None

        lx, ly = self.rect.to_local(x, y)

        for bid, _label, r in self.buttons:
            if r.contains(lx, ly):
                return bid

        return None

    def poll_key(self) -> int:
        if not self._open:
            return -1

        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(self.NAME)
            self._open = False

    # -- drawing --------------------------------------------------------------
    def render(self, lines: list[str], game_view: np.ndarray | None, cam_view: np.ndarray | None,
               dwell_target: str | None, dwell_progress: float, gaze_local_hud: tuple[float, float] | None = None) -> np.ndarray:
        c = self._canvas
        c[:] = BG

        # Status text block.
        y = 22

        for i, line in enumerate(lines[:6]):
            cv2.putText(c, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT if i < 2 else DIM, 1, cv2.LINE_AA)
            y += 20

        # Camera thumbnail, top right.
        if cam_view is not None and self.cfg.hud.show_camera:
            tw, th = 150, int(150 * cam_view.shape[0] / max(1, cam_view.shape[1]))
            thumb = cv2.resize(cam_view, (tw, th))
            c[8:8 + th, self.w - tw - 8:self.w - 8] = thumb

        # Game view, middle.
        top = 132
        view_h = self.buttons[0][2].top - top - 10

        if game_view is not None:
            gh, gw = game_view.shape[:2]
            s = min((self.w - 16) / gw, view_h / gh)
            vw, vh = max(1, int(gw * s)), max(1, int(gh * s))
            small = cv2.resize(game_view, (vw, vh), interpolation = cv2.INTER_AREA)
            x0 = (self.w - vw) // 2
            c[top:top + vh, x0:x0 + vw] = small
            cv2.rectangle(c, (x0 - 1, top - 1), (x0 + vw, top + vh), PANEL, 1)
        else:
            cv2.rectangle(c, (8, top), (self.w - 8, top + view_h), PANEL, -1)
            cv2.putText(c, "game window not found", (20, top + view_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DIM, 1, cv2.LINE_AA)

        # Dwell buttons.
        for bid, label, r in self.buttons:
            active = bid == dwell_target
            cv2.rectangle(c, (r.left, r.top), (r.right, r.bottom), PANEL, -1)

            if active and dwell_progress > 0:
                fill = int(r.width * dwell_progress)
                cv2.rectangle(c, (r.left, r.top), (r.left + fill, r.bottom), FIRE if dwell_progress >= 1 else ACCENT, -1)

            cv2.rectangle(c, (r.left, r.top), (r.right, r.bottom), ACCENT if active else DIM, 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(c, label, (r.left + (r.width - tw) // 2, r.top + (r.height + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)

        # Gaze cursor when the player is looking at the HUD itself.
        if gaze_local_hud is not None:
            gx, gy = int(gaze_local_hud[0]), int(gaze_local_hud[1])

            if 0 <= gx < self.w and 0 <= gy < self.h:
                cv2.circle(c, (gx, gy), 7, (255, 0, 255), 2)

        if self._open:
            cv2.imshow(self.NAME, c)

        return c


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    from .config import default_config
    from .screen_reader import synthetic_battle_frame

    hud = Hud(default_config())
    # Buttons tile the bottom row without overlapping.
    rects = [r for _, _, r in hud.buttons]

    for a, b in zip(rects, rects[1:]):
        assert a.right < b.left

    # Hit testing uses the window's screen rectangle.
    hud.rect = Rect(1000, 200, hud.w, hud.h)
    r0 = rects[0]
    assert hud.hit_test(1000 + r0.left + 5, 200 + r0.top + 5) == "confirm"
    assert hud.hit_test(1000 + 5, 200 + 5) is None
    assert hud.hit_test(0, 0) is None
    img = hud.render(["mode: dodge:red", "held: right", "fps 30/30/60"], synthetic_battle_frame(), None, "confirm", 0.5)
    assert img.shape == (hud.h, hud.w, 3) and img.any()
    print("hud: OK")


if __name__ == "__main__":
    _self_test()
