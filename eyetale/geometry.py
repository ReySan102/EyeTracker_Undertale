"""
Rectangles and coordinate conversion.

Everything in the project is expressed in one of two frames:
  * screen coordinates  - absolute pixels on the desktop (gaze, mouse, windows)
  * window-local        - pixels relative to the top-left of the game's client area

Rect is the one type that moves between them.

Run:  python -m eyetale.geometry
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen = True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width / 2.0, self.top + self.height / 2.0

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        """Screen -> window-local."""
        return x - self.left, y - self.top

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Window-local -> screen."""
        return x + self.left, y + self.top

    def offset(self, dx: int, dy: int) -> "Rect":
        return Rect(self.left + dx, self.top + dy, self.width, self.height)

    def shrink(self, px: int) -> "Rect":
        return Rect(self.left + px, self.top + px, max(0, self.width - 2 * px), max(0, self.height - 2 * px))

    def clamp_point(self, x: float, y: float) -> tuple[float, float]:
        return (min(max(x, self.left), self.right - 1), min(max(y, self.top), self.bottom - 1))

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.width, self.height


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    r = Rect(100, 50, 640, 480)
    assert r.right == 740 and r.bottom == 530
    assert r.center == (420.0, 290.0)
    assert r.contains(100, 50) and not r.contains(740, 50)
    assert r.to_local(420, 290) == (320.0, 240.0)
    assert r.to_screen(320, 240) == (420.0, 290.0)
    assert r.shrink(5) == Rect(105, 55, 630, 470)
    assert r.clamp_point(-10, 9999) == (100, 529)
    assert abs(distance(0, 0, 3, 4) - 5.0) < 1e-9
    print("geometry: OK")


if __name__ == "__main__":
    _self_test()
