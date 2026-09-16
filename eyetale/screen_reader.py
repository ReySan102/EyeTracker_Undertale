"""
Read the game's state straight off the screen.

Undertale exposes no API, so the program looks at the pixels of the game
window the same way the player does. Three questions are answered per frame:

  1. Are we in a battle?      four orange/yellow button outlines along the
                              bottom edge (FIGHT / ACT / ITEM / MERCY)
  2. Where is the bullet box? the largest white-bordered rectangle with a dark
                              interior
  3. Where is the SOUL?       the biggest blob of SOUL-coloured pixels inside
                              the box; its colour says which SOUL mode is active

All detection functions are pure (numpy array in, dataclass out) and are
tested on synthetic frames below. `ScreenReader` is the only piece that
touches the screen (via mss). Pixel constants are for the default 640x480
window and are scaled by `frame_width / 640` for larger windows.

Run:  python -m eyetale.screen_reader
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .config import GameConfig
from .geometry import Rect

BASE_W = 640.0

# BGR colours (OpenCV order). Tolerances are per channel.
SOUL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 192, 0),
    "yellow": (0, 255, 255),
    "purple": (217, 53, 213),
}
UI_ORANGE_BGR = (0, 128, 255)
UI_YELLOW_BGR = (0, 255, 255)
BORDER_PX = 5          # Thickness of the bullet box border at 640x480


@dataclass
class GameState:
    in_battle: bool = False
    box: Rect | None = None              # Bullet box interior, window-local pixels
    soul: tuple[float, float] | None = None   # SOUL centre, window-local pixels
    soul_color: str | None = None
    scale: float = 1.0
    frame_w: int = 0
    frame_h: int = 0

    @property
    def soul_in_box(self) -> bool:
        return self.box is not None and self.soul is not None

    @property
    def mode_name(self) -> str:
        if not self.in_battle:
            return "overworld"

        if self.soul_in_box:
            return f"dodge:{self.soul_color}"

        return "battle-menu"


# ---------------------------------------------------------------------------
# Pure detection
# ---------------------------------------------------------------------------

def color_mask(frame: np.ndarray, bgr: tuple[int, int, int], tol: int) -> np.ndarray:
    lo = np.array([max(0, c - tol) for c in bgr], dtype = np.uint8)
    hi = np.array([min(255, c + tol) for c in bgr], dtype = np.uint8)

    return cv2.inRange(frame, lo, hi)


def detect_battle_ui(frame: np.ndarray, scale: float, tol: int = 40) -> bool:
    """True when at least three button-sized orange/yellow outlines sit in the bottom 16 % of the window."""
    h, w = frame.shape[:2]
    band_top = int(h * 0.84)
    band = frame[band_top:, :]
    mask = cv2.bitwise_or(color_mask(band, UI_ORANGE_BGR, tol), color_mask(band, UI_YELLOW_BGR, tol))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits = 0

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        if 70 * scale <= bw <= 170 * scale and 25 * scale <= bh <= 70 * scale and 1.6 <= bw / max(1, bh) <= 4.5:
            hits += 1

    return hits >= 3


def find_bullet_box(frame: np.ndarray, scale: float) -> Rect | None:
    """Largest white-bordered rectangle with a dark interior. Returns the interior (border removed) in window-local pixels."""
    h, w = frame.shape[:2]
    white = cv2.inRange(frame, np.array([225, 225, 225], np.uint8), np.array([255, 255, 255], np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    min_side = 36 * scale

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        if bw < min_side or bh < min_side or y + bh > h * 0.93:
            continue

        rect_area = bw * bh

        if rect_area <= best_area or rect_area < 0.012 * w * h:
            continue

        if cv2.contourArea(c) / rect_area < 0.80:          # Must trace a full rectangle
            continue

        b = int(round(BORDER_PX * scale))
        inner = frame[y + b:y + bh - b, x + b:x + bw - b]

        if inner.size == 0 or inner.mean() > 110:           # Interior must be mostly black
            continue

        best, best_area = Rect(x + b, y + b, bw - 2 * b, bh - 2 * b), rect_area

    return best


def find_soul(frame: np.ndarray, region: Rect, min_pixels: int, tol: int) -> tuple[float, float, str] | None:
    """Biggest SOUL-coloured blob inside `region`. Returns (x, y, colour) in window-local pixels, or None."""
    sub = frame[region.top:region.bottom, region.left:region.right]

    if sub.size == 0:
        return None

    best = None

    for name, bgr in SOUL_COLORS_BGR.items():
        mask = color_mask(sub, bgr, tol)
        count = int(cv2.countNonZero(mask))

        if count >= min_pixels and (best is None or count > best[0]):
            m = cv2.moments(mask, binaryImage = True)

            if m["m00"] > 0:
                best = (count, m["m10"] / m["m00"] + region.left, m["m01"] / m["m00"] + region.top, name)

    if best is None:
        return None

    return best[1], best[2], best[3]


def classify_state(frame: np.ndarray, cfg: GameConfig, fallback_box: Rect | None = None) -> GameState:
    """
    Everything the mapper needs from one frame. `fallback_box` is the box from a recent frame, used when a busy attack hides the border for a moment.
    """
    h, w = frame.shape[:2]
    scale = w / BASE_W
    st = GameState(scale = scale, frame_w = w, frame_h = h)
    st.in_battle = detect_battle_ui(frame, scale, tol = min(cfg.color_tolerance, 40))

    if not st.in_battle:
        return st

    st.box = find_bullet_box(frame, scale) or fallback_box

    if st.box is not None:
        hit = find_soul(frame, st.box, int(cfg.soul_min_pixels * scale * scale), cfg.color_tolerance)

        if hit is not None:
            st.soul, st.soul_color = (hit[0], hit[1]), hit[2]

    return st


class StateMemory:
    """
    Short-term memory over successive GameStates.

    Two things flicker in the real game: the box border can be hidden by a
    dense attack for a frame or two, and the SOUL blinks after taking damage
    (invincibility frames). Losing either for a single frame would drop the
    mapper out of dodge mode and release the arrow keys at the worst moment,
    so the last good box / SOUL is carried forward for a short time.
    """

    def __init__(self, box_memory_s: float = 0.5, soul_memory_s: float = 0.25) -> None:
        self.box_memory_s, self.soul_memory_s = box_memory_s, soul_memory_s
        self.last_box: Rect | None = None
        self.last_box_t = -1e9
        self.last_soul: tuple[float, float] | None = None
        self.last_soul_color: str | None = None
        self.last_soul_t = -1e9

    def fallback_box(self, now: float) -> Rect | None:
        return self.last_box if now - self.last_box_t <= self.box_memory_s else None

    def update(self, st: GameState, now: float) -> GameState:
        if not st.in_battle:
            self.last_box = self.last_soul = None

            return st

        if st.box is not None:
            self.last_box, self.last_box_t = st.box, now

        if st.soul is not None:
            self.last_soul, self.last_soul_color, self.last_soul_t = st.soul, st.soul_color, now
        elif st.box is not None and self.last_soul is not None and now - self.last_soul_t <= self.soul_memory_s:
            st.soul, st.soul_color = self.last_soul, self.last_soul_color

        return st


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

class ScreenReader:
    """Grabs the game window's client area as a BGR numpy array. Create and use from one thread (mss instances are thread-bound on Windows)."""

    def __init__(self) -> None:
        import mss

        self._sct = mss.mss()

    def grab(self, rect: Rect) -> np.ndarray:
        shot = self._sct.grab({"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height})

        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)   # Contiguous BGR copy

    def close(self) -> None:
        self._sct.close()


def draw_state(frame: np.ndarray, st: GameState, gaze_local: tuple[float, float] | None = None, dead_zone_px: float | None = None) -> np.ndarray:
    """Annotated copy of the frame for the HUD / watch mode."""
    out = frame.copy()

    if st.box is not None:
        b = st.box
        cv2.rectangle(out, (b.left, b.top), (b.right, b.bottom), (0, 255, 0), 1)

    if st.soul is not None:
        sx, sy = int(st.soul[0]), int(st.soul[1])
        cv2.circle(out, (sx, sy), 10, (255, 255, 0), 1)

        if dead_zone_px:
            cv2.circle(out, (sx, sy), int(dead_zone_px), (0, 200, 200), 1)

    if gaze_local is not None:
        gx, gy = int(gaze_local[0]), int(gaze_local[1])
        cv2.circle(out, (gx, gy), 6, (255, 0, 255), 2)
        cv2.line(out, (gx - 12, gy), (gx + 12, gy), (255, 0, 255), 1)
        cv2.line(out, (gx, gy - 12), (gx, gy + 12), (255, 0, 255), 1)

    cv2.putText(out, st.mode_name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out


# ---------------------------------------------------------------------------
# Self-test: synthetic frames that imitate the real UI layout
# ---------------------------------------------------------------------------

def synthetic_battle_frame(soul_xy = (320, 330), soul_color = "red", box = (32, 250, 576, 150),
                           with_box: bool = True, size = (640, 480)) -> np.ndarray:
    w, h = size
    f = np.zeros((h, w, 3), np.uint8)

    # Four button outlines along the bottom, the first one selected (yellow)
    for i, x in enumerate((32, 185, 345, 500)):
        color = UI_YELLOW_BGR if i == 0 else UI_ORANGE_BGR
        cv2.rectangle(f, (x, 432), (x + 110, 474), color, 2)
        cv2.putText(f, "TXT", (x + 30, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # HP bar: yellow on red (outside the box, must not be mistaken for the SOUL)
    cv2.rectangle(f, (275, 400), (335, 412), (0, 0, 255), -1)
    cv2.rectangle(f, (275, 400), (300, 412), (0, 255, 255), -1)
    # A white enemy sprite (blob) at the top, to make sure it is not taken for the box
    cv2.ellipse(f, (320, 140), (60, 70), 0, 0, 360, (255, 255, 255), -1)

    if with_box:
        bx, by, bw, bh = box
        cv2.rectangle(f, (bx, by), (bx + bw, by + bh), (255, 255, 255), BORDER_PX)
        cv2.putText(f, "* Froggit hops close!", (bx + 20, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if soul_xy is not None:
            sx, sy = soul_xy
            cv2.rectangle(f, (sx - 8, sy - 8), (sx + 8, sy + 8), SOUL_COLORS_BGR[soul_color], -1)

    return f


def synthetic_overworld_frame() -> np.ndarray:
    f = np.full((480, 640, 3), (90, 70, 60), np.uint8)          # Brownish room
    cv2.rectangle(f, (0, 300), (640, 480), (40, 110, 200), -1)  # Orange-ish floor (Hotland-like)
    cv2.rectangle(f, (32, 320), (608, 460), (0, 0, 0), -1)      # Dialogue box
    cv2.rectangle(f, (32, 320), (608, 460), (255, 255, 255), 5)
    cv2.putText(f, "* Hello.", (60, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return f


def _self_test() -> None:
    cfg = GameConfig()
    st = classify_state(synthetic_battle_frame(), cfg)
    assert st.in_battle, "battle UI not detected"
    # cv2.rectangle centres its 5 px border on the given coordinates, so the
    # interior starts about 3 px inside (32 -> ~35, 250 -> ~253).
    assert st.box is not None and abs(st.box.left - 35) <= 3 and abs(st.box.top - 253) <= 3, st.box
    assert st.soul is not None and abs(st.soul[0] - 320) < 1.5 and abs(st.soul[1] - 330) < 1.5, st.soul
    assert st.soul_color == "red" and st.mode_name == "dodge:red"

    st_blue = classify_state(synthetic_battle_frame(soul_color = "blue"), cfg)
    assert st_blue.soul_color == "blue"

    # Menu phase: box present, SOUL outside it (on the buttons) -> battle-menu.
    st_menu = classify_state(synthetic_battle_frame(soul_xy = None), cfg)
    assert st_menu.in_battle and st_menu.box is not None and st_menu.soul is None
    assert st_menu.mode_name == "battle-menu"

    # Small attack box, scaled 2x window.
    big = cv2.resize(synthetic_battle_frame(soul_xy = (320, 340), box = (260, 280, 120, 120)), (1280, 960),
                     interpolation = cv2.INTER_NEAREST)
    st_big = classify_state(big, cfg)
    assert st_big.in_battle and st_big.box is not None and st_big.soul is not None
    assert abs(st_big.soul[0] - 640) < 3 and abs(st_big.soul[1] - 680) < 3, st_big.soul

    # Overworld with a white-bordered dialogue box and orange floor: no battle.
    st_ow = classify_state(synthetic_overworld_frame(), cfg)
    assert not st_ow.in_battle and st_ow.mode_name == "overworld"

    out = draw_state(synthetic_battle_frame(), st, gaze_local = (300, 300), dead_zone_px = 14)
    assert out.shape == (480, 640, 3)

    # StateMemory: a frame where the SOUL blinks out keeps the previous SOUL briefly.
    mem = StateMemory()
    s1 = mem.update(classify_state(synthetic_battle_frame(), cfg), now = 0.0)
    assert s1.soul is not None
    blank = classify_state(synthetic_battle_frame(soul_xy = None), cfg, fallback_box = mem.fallback_box(0.1))
    s2 = mem.update(blank, now = 0.1)
    assert s2.soul is not None and s2.soul_color == "red"            # Carried forward
    s3 = mem.update(classify_state(synthetic_battle_frame(soul_xy = None), cfg), now = 1.0)
    assert s3.soul is None                                            # Memory expired
    assert mem.fallback_box(1.2) is not None and mem.fallback_box(5.0) is None
    print("screen_reader: OK")


if __name__ == "__main__":
    _self_test()
