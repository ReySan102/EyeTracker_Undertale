"""
All operating-system specific calls in one place, each with a harmless
fallback so the rest of the project imports and self-tests on any OS.

Windows implementations use ctypes (user32) and pywin32 (win32gui).

  screen_size()             primary monitor size in physical pixels
  set_dpi_aware()           make coordinates physical pixels even at 125 % scaling
  cursor_pos()              mouse position (used by the mouse gaze source)
  mouse_button_down(n)      is mouse button n held (1 = left, 2 = right)
  find_window_rect(title)   client-area Rect of a top-level window, in screen px
  foreground_title()        title of the window that has keyboard focus
  focus_window(title)       try to bring a window to the foreground

Run:  python -m eyetale.winapi
"""
from __future__ import annotations

import sys

from .geometry import Rect

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32

    try:
        import win32gui  # pywin32
    except ImportError:  # pragma: no cover
        win32gui = None
else:
    win32gui = None


def set_dpi_aware() -> None:
    """Call once at start-up. Without this, Windows lies to Python about pixel coordinates on scaled displays and gaze zones would not line up."""
    if not IS_WINDOWS:
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor v2 where available
    except Exception:
        try:
            _user32.SetProcessDPIAware()
        except Exception:
            pass


def screen_size() -> tuple[int, int]:
    if IS_WINDOWS:
        return int(_user32.GetSystemMetrics(0)), int(_user32.GetSystemMetrics(1))

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()

        return int(w), int(h)
    except Exception:
        return 1920, 1080


def cursor_pos() -> tuple[int, int]:
    if IS_WINDOWS:
        pt = wintypes.POINT()
        _user32.GetCursorPos(ctypes.byref(pt))

        return int(pt.x), int(pt.y)

    w, h = screen_size()

    return w // 2, h // 2


def mouse_button_down(button: int = 1) -> bool:
    """1 = left, 2 = right, 3 = middle."""
    if not IS_WINDOWS:
        return False

    vk = {1: 0x01, 2: 0x02, 3: 0x04}.get(button, 0x01)

    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def _find_hwnd(title: str):
    if win32gui is None:
        return None

    hwnd = win32gui.FindWindow(None, title)

    return hwnd or None


def find_window_rect(title: str) -> Rect | None:
    """
    Client-area rectangle of the window titled `title`, in screen pixels. Returns None when the window does not exist (or on non-Windows systems).
    """
    hwnd = _find_hwnd(title)

    if not hwnd:
        return None

    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        sx, sy = win32gui.ClientToScreen(hwnd, (left, top))
        w, h = right - left, bottom - top

        if w <= 0 or h <= 0:
            return None

        return Rect(int(sx), int(sy), int(w), int(h))
    except Exception:
        return None


def foreground_title() -> str:
    if win32gui is None:
        return ""

    try:
        return win32gui.GetWindowText(win32gui.GetForegroundWindow())
    except Exception:
        return ""


def focus_window(title: str) -> bool:
    hwnd = _find_hwnd(title)

    if not hwnd:
        return False

    try:
        win32gui.SetForegroundWindow(hwnd)

        return True
    except Exception:
        return False


def move_window(title: str, x: int, y: int) -> bool:
    """Move a top-level window (used to park the HUD next to the game)."""
    hwnd = _find_hwnd(title)

    if not hwnd:
        return False

    try:
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        win32gui.MoveWindow(hwnd, int(x), int(y), r - l, b - t, True)

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    w, h = screen_size()
    assert w > 0 and h > 0
    x, y = cursor_pos()
    assert 0 <= x <= w * 4 and 0 <= y <= h * 4   # Multi-monitor setups can exceed the primary size
    assert find_window_rect("this window does not exist 12345") is None
    assert isinstance(foreground_title(), str)
    assert mouse_button_down(1) in (True, False)
    print(f"winapi: OK (platform={'windows' if IS_WINDOWS else 'fallback'}, screen={w}x{h})")


if __name__ == "__main__":
    _self_test()
