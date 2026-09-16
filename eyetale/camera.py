"""
Webcam capture on a background thread.

OpenCV's VideoCapture.read() blocks for a whole frame period, so it must not
run inside the 60 Hz control loop. The thread keeps only the newest frame;
if the consumer is slower than the camera, old frames are dropped rather than
queued, which keeps latency at one frame instead of growing without bound.

Run:  python -m eyetale.camera         (opens camera 0 for two seconds if one exists)
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class CameraThread:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._cap = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._t: float = 0.0
        self._seq: int = 0
        self._running = False
        self.measured_fps: float = 0.0

    def start(self) -> bool:
        if cv2 is None:
            print("[camera] OpenCV not installed")

            return False

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.index, backend)

        if not self._cap.isOpened():
            print(f"[camera] could not open camera {self.index}")

            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Do not queue stale frames
        self._running = True
        self._thread = threading.Thread(target = self._loop, name = "camera", daemon = True)
        self._thread.start()

        return True

    def _loop(self) -> None:
        n, t0 = 0, time.perf_counter()

        while self._running:
            ok, frame = self._cap.read()

            if not ok:
                time.sleep(0.005)
                continue

            now = time.perf_counter()

            with self._lock:
                self._frame, self._t = frame, now
                self._seq += 1

            n += 1

            if now - t0 >= 1.0:
                self.measured_fps, n, t0 = n / (now - t0), 0, now

    def latest(self) -> tuple[np.ndarray, float, int] | None:
        """Newest frame, its capture time (perf_counter seconds) and a sequence number so consumers can skip frames they have already processed."""
        with self._lock:
            if self._frame is None:
                return None

            return self._frame, self._t, self._seq

    def stop(self) -> None:
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout = 1.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    cam = CameraThread(0, 640, 480, 30)

    if not cam.start():
        print("camera: SKIP (no camera available here)")

        return

    time.sleep(2.0)
    got = cam.latest()
    cam.stop()
    assert got is not None, "no frame in two seconds"
    frame, t, seq = got
    assert frame.ndim == 3 and seq > 0
    print(f"camera: OK ({frame.shape[1]}x{frame.shape[0]}, ~{cam.measured_fps:.0f} fps)")


if __name__ == "__main__":
    _self_test()
