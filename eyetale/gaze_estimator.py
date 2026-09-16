"""
Gaze sources: anything that can answer "where is the player looking right now?"

    GazeSample          one reading: screen x/y, validity, eye-openness (EAR)
    GazeSource          the interface the control loop talks to
    WebcamGazeSource    camera -> FaceTracker -> CalibrationModel -> One Euro filter
    MouseGazeSource     the mouse cursor pretends to be the gaze; hold the left
                        button to "close your eyes". Lets you develop and test
                        everything downstream before the camera side works.
    ReplayGazeSource    plays back a CSV log, for repeatable tests
    GazeLogger          writes samples to CSV so a session can be replayed

Every source runs its own thread (or none) and exposes `latest()`, which
never blocks. The control loop polls it at its own rate.

Run:  python -m eyetale.gaze_estimator
"""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .calibration import CalibrationModel
from .config import AppConfig
from .face_tracker import FaceFeatures
from .filters import OneEuroFilter2D
from . import winapi


@dataclass
class GazeSample:
    t: float                    # perf_counter seconds when the frame was captured
    x: float                    # Screen pixels (filtered)
    y: float
    valid: bool                 # Face found and model available
    ear: float                  # min(eye aspect ratio of both eyes); 0 when eyes are shut
    raw_x: float = 0.0          # Before filtering (debugging)
    raw_y: float = 0.0
    seq: int = 0                # Increments per new camera frame


class GazeSource:
    def start(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def latest(self) -> GazeSample | None:
        raise NotImplementedError

    def latest_features(self) -> FaceFeatures | None:
        return None

    def latest_frame(self):
        return None

    @property
    def ear_baseline(self) -> float:
        return 0.0

    def set_model(self, model: CalibrationModel | None) -> None:
        pass


# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------

class WebcamGazeSource(GazeSource):
    def __init__(self, cfg: AppConfig, model: CalibrationModel | None = None) -> None:
        self.cfg = cfg
        self.model = model
        self._camera = None
        self._tracker = None
        self._filter = OneEuroFilter2D(cfg.filter.min_cutoff, cfg.filter.beta, cfg.filter.d_cutoff)
        self._lock = threading.Lock()
        self._sample: GazeSample | None = None
        self._features: FaceFeatures | None = None
        self._frame = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.fps: float = 0.0

    def start(self) -> bool:
        from .camera import CameraThread
        from .face_tracker import FaceTracker

        self._camera = CameraThread(self.cfg.camera.index, self.cfg.camera.width, self.cfg.camera.height, self.cfg.camera.fps)

        if not self._camera.start():
            return False

        self._tracker = FaceTracker(self.cfg.model_path)
        self._running = True
        self._thread = threading.Thread(target = self._loop, name = "gaze", daemon = True)
        self._thread.start()

        return True

    def set_model(self, model: CalibrationModel | None) -> None:
        with self._lock:
            self.model = model
            self._filter.reset()

    @property
    def ear_baseline(self) -> float:
        return self.model.ear_baseline if self.model else 0.0

    def _loop(self) -> None:
        last_seq = -1
        n, t0 = 0, time.perf_counter()

        while self._running:
            got = self._camera.latest()

            if got is None or got[2] == last_seq:
                time.sleep(0.002)
                continue

            frame, t, seq = got
            last_seq = seq
            f = self._tracker.process(frame, t, seq)

            with self._lock:
                model = self.model

            if f.ok and model is not None:
                rx, ry = model.predict_features(f)
                fx, fy = self._filter.filter(rx, ry, t)
                sample = GazeSample(t, fx, fy, True, f.ear, rx, ry, seq)
            else:
                prev = self._sample
                x, y = (prev.x, prev.y) if prev else (0.0, 0.0)
                sample = GazeSample(t, x, y, False, f.ear if f.ok else 0.0, x, y, seq)

            with self._lock:
                self._sample, self._features, self._frame = sample, f, frame

            n += 1
            now = time.perf_counter()

            if now - t0 >= 1.0:
                self.fps, n, t0 = n / (now - t0), 0, now

    def latest(self) -> GazeSample | None:
        with self._lock:
            return self._sample

    def latest_features(self) -> FaceFeatures | None:
        with self._lock:
            return self._features

    def latest_frame(self):
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._running = False

        if self._thread:
            self._thread.join(timeout = 1.0)

        if self._camera:
            self._camera.stop()

        if self._tracker:
            self._tracker.close()


# ---------------------------------------------------------------------------
# Mouse stand-in
# ---------------------------------------------------------------------------

class MouseGazeSource(GazeSource):
    """Cursor = gaze. Left mouse button held = eyes closed (EAR 0)."""

    OPEN_EAR = 0.30

    def __init__(self) -> None:
        self._seq = 0

    @property
    def ear_baseline(self) -> float:
        return self.OPEN_EAR

    def latest(self) -> GazeSample | None:
        x, y = winapi.cursor_pos()
        self._seq += 1
        ear = 0.0 if winapi.mouse_button_down(1) else self.OPEN_EAR

        return GazeSample(time.perf_counter(), float(x), float(y), True, ear, float(x), float(y), self._seq)


# ---------------------------------------------------------------------------
# Replay and logging
# ---------------------------------------------------------------------------

LOG_FIELDS = ["t", "x", "y", "valid", "ear", "raw_x", "raw_y", "seq"]


class GazeLogger:
    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents = True, exist_ok = True)
        self._fh = p.open("w", newline = "", encoding = "utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(LOG_FIELDS)
        self._last_seq = -1

    def log(self, s: GazeSample | None) -> None:
        if s is None or s.seq == self._last_seq:
            return

        self._last_seq = s.seq
        self._w.writerow([f"{s.t:.4f}", f"{s.x:.1f}", f"{s.y:.1f}", int(s.valid), f"{s.ear:.3f}",
                          f"{s.raw_x:.1f}", f"{s.raw_y:.1f}", s.seq])

    def close(self) -> None:
        self._fh.close()


class ReplayGazeSource(GazeSource):
    """
    Replays a CSV written by GazeLogger.

    realtime=True  : latest() returns the sample whose timestamp matches wall-clock time
    realtime=False : latest() returns the next sample each call (fast, deterministic tests)
    """

    def __init__(self, path: str | Path, realtime: bool = True) -> None:
        self.samples: list[GazeSample] = []

        with Path(path).open(encoding = "utf-8") as fh:
            for row in csv.DictReader(fh):
                self.samples.append(GazeSample(float(row["t"]), float(row["x"]), float(row["y"]),
                                               row["valid"] in ("1", "True"), float(row["ear"]),
                                               float(row["raw_x"]), float(row["raw_y"]), int(row["seq"])))

        self.realtime = realtime
        self._i = 0
        self._t_start = None

    @property
    def ear_baseline(self) -> float:
        return 0.30

    @property
    def finished(self) -> bool:
        if not self.samples:
            return True

        if not self.realtime:
            return self._i >= len(self.samples)

        return self._t_start is not None and (time.perf_counter() - self._t_start) > self.samples[-1].t + 0.5

    def latest(self) -> GazeSample | None:
        if not self.samples:
            return None

        if not self.realtime:
            if self._i >= len(self.samples):
                return self.samples[-1]

            s = self.samples[self._i]
            self._i += 1

            return s

        now = time.perf_counter()

        if self._t_start is None:
            self._t_start = now - self.samples[0].t

        target = now - self._t_start

        while self._i + 1 < len(self.samples) and self.samples[self._i + 1].t <= target:
            self._i += 1

        return self.samples[self._i]


def make_gaze_source(cfg: AppConfig, model: CalibrationModel | None, replay_path: str | None = None) -> GazeSource:
    if cfg.gaze_source == "mouse":
        return MouseGazeSource()

    if cfg.gaze_source == "replay":
        return ReplayGazeSource(replay_path or cfg.log_path)

    return WebcamGazeSource(cfg, model)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import tempfile

    m = MouseGazeSource()
    s = m.latest()
    assert s is not None and s.valid and s.ear in (0.0, MouseGazeSource.OPEN_EAR)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "log.csv"
        log = GazeLogger(p)

        for i in range(5):
            log.log(GazeSample(i * 0.1, 100 + i, 200, True, 0.3, 100 + i, 200, i))

        log.log(GazeSample(0.5, 0, 0, True, 0.3, 0, 0, 4))    # Duplicate seq: skipped
        log.close()
        rp = ReplayGazeSource(p, realtime = False)
        assert len(rp.samples) == 5
        xs = [rp.latest().x for _ in range(6)]
        assert xs == [100, 101, 102, 103, 104, 104] and rp.finished
        rp2 = ReplayGazeSource(p, realtime = True)
        assert rp2.latest().x == 100

    print("gaze_estimator: OK")


if __name__ == "__main__":
    _self_test()
