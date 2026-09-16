"""
Calibration: learn the mapping from face features to screen pixels.

The webcam does not know where you are looking. It knows where your irises
are relative to your eye corners and how your head is posed. Calibration
shows a dot at known screen positions, records the features while you stare
at each one, and fits a ridge regression from features to pixel coordinates.
The fitted model is saved to calibration.json and loaded at play time.

Also measured here: the open-eye EAR baseline, which sets the blink
thresholds for this particular person and camera.

Pieces:
  calibration_points()  Where the dots go (fractions of the screen)
  fit_model()           Standardise features, closed-form ridge solve
  CalibrationModel      Predict / save / load
  run_calibration()     The on-screen routine (OpenCV full-screen window)

Run:  python -m eyetale.calibration      (self-test with synthetic data)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .face_tracker import FEATURE_DIM, FaceFeatures, feature_vector


def calibration_points(extra: bool = True, margin: float = 0.08) -> list[tuple[float, float]]:
    """3x3 grid, optionally with four inner points (13 total) for a better polynomial fit. Values are fractions of screen width / height."""
    xs = [margin, 0.5, 1.0 - margin]
    ys = [margin, 0.5, 1.0 - margin]
    pts = [(x, y) for y in ys for x in xs]

    if extra:
        pts += [(0.27, 0.27), (0.73, 0.27), (0.27, 0.73), (0.73, 0.73)]

    return pts


@dataclass
class CalibrationModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray                 # (FEATURE_DIM, 2)
    screen_w: int
    screen_h: int
    ear_baseline: float = 0.0           # Open-eye EAR measured during calibration
    rmse_px: float = 0.0
    point_errors_px: list[float] = field(default_factory = list)
    created: str = ""

    def predict(self, vec: np.ndarray) -> tuple[float, float]:
        z = (vec - self.mean) / self.std
        x, y = z @ self.weights

        return float(x), float(y)

    def predict_features(self, f: FaceFeatures) -> tuple[float, float]:
        return self.predict(feature_vector(f))

    def save(self, path: str | Path) -> None:
        data = {
            "mean": self.mean.tolist(), "std": self.std.tolist(), "weights": self.weights.tolist(),
            "screen_w": self.screen_w, "screen_h": self.screen_h, "ear_baseline": self.ear_baseline,
            "rmse_px": self.rmse_px, "point_errors_px": self.point_errors_px, "created": self.created,
            "feature_dim": FEATURE_DIM,
        }
        Path(path).write_text(json.dumps(data, indent = 1), encoding = "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationModel | None":
        p = Path(path)

        if not p.exists():
            return None

        d = json.loads(p.read_text(encoding = "utf-8"))

        if d.get("feature_dim") != FEATURE_DIM:
            print("[calibration] saved model has a different feature layout; re-run calibrate")

            return None

        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["weights"]), int(d["screen_w"]), int(d["screen_h"]),
                   float(d.get("ear_baseline", 0.0)), float(d.get("rmse_px", 0.0)), list(d.get("point_errors_px", [])), d.get("created", ""))


def fit_model(X: np.ndarray, Y: np.ndarray, screen_w: int, screen_h: int, lam: float = 1.0, ear_baseline: float = 0.0) -> CalibrationModel:
    """
    Ridge regression  W = (Z'Z + lam * I)^(-1) Z'Y  on standardised features Z.

    X: (n, FEATURE_DIM) feature vectors, Y: (n, 2) target screen pixels.
    The bias column (constant 1) has zero variance, so its std is forced to 1 and it is excluded from the penalty.
    """
    X = np.asarray(X, dtype = np.float64)
    Y = np.asarray(Y, dtype = np.float64)
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    std[std < 1e-9] = 1.0
    mean[0], std[0] = 0.0, 1.0            # Keep the bias column as a plain 1
    Z = (X - mean) / std
    penalty = lam * np.eye(Z.shape[1])
    penalty[0, 0] = 0.0
    W = np.linalg.solve(Z.T @ Z + penalty, Z.T @ Y)
    pred = Z @ W
    rmse = float(np.sqrt(np.mean(np.sum((pred - Y) ** 2, axis = 1))))

    return CalibrationModel(mean, std, W, screen_w, screen_h, ear_baseline, rmse, created = time.strftime("%Y-%m-%d %H:%M:%S"))


def evaluate_per_point(model: CalibrationModel, X_by_point: list[np.ndarray], targets: list[tuple[float, float]]) -> list[float]:
    """Mean error in pixels for each calibration point (for the results screen)."""
    errs = []

    for X, (tx, ty) in zip(X_by_point, targets):
        if len(X) == 0:
            errs.append(float("nan"))
            continue

        pred = np.array([model.predict(v) for v in X])
        errs.append(float(np.mean(np.hypot(pred[:, 0] - tx, pred[:, 1] - ty))))

    return errs


# ---------------------------------------------------------------------------
# On-screen routine
# ---------------------------------------------------------------------------

def run_calibration(feature_provider: Callable[[], FaceFeatures | None], screen_w: int, screen_h: int, settle_s: float = 1.0, collect_s: float = 1.2,
                    points: list[tuple[float, float]] | None = None, window_name: str = "EyeTale Calibration") -> CalibrationModel | None:
    """Show dots, collect features, fit. `feature_provider` returns the newest FaceFeatures (or None). Press Esc to abort. Returns None on abort."""
    import cv2

    pts = points or calibration_points()
    targets = [(fx * screen_w, fy * screen_h) for fx, fy in pts]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas = np.zeros((screen_h, screen_w, 3), dtype = np.uint8)

    def show(msg: str, target = None, progress: float = 0.0, extra = None) -> bool:
        canvas[:] = (20, 20, 20)

        if extra is not None:
            extra(canvas)

        if target is not None:
            x, y = int(target[0]), int(target[1])
            cv2.circle(canvas, (x, y), 22, (80, 80, 80), 2)
            cv2.circle(canvas, (x, y), int(22 - 16 * progress), (0, 220, 255), -1)
            cv2.circle(canvas, (x, y), 3, (0, 0, 0), -1)

        cv2.putText(canvas, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow(window_name, canvas)

        return cv2.waitKey(1) != 27           # False on Esc

    # Intro screen -----------------------------------------------------------
    t_end = time.perf_counter() + 2.5

    while time.perf_counter() < t_end:
        if not show("Keep your head still. Follow the dot with your eyes. Esc aborts."):
            cv2.destroyWindow(window_name)

            return None

    X_by_point: list[np.ndarray] = []
    ears: list[float] = []

    for i, (target, frac) in enumerate(zip(targets, pts)):
        samples: list[np.ndarray] = []
        t0 = time.perf_counter()
        last_seq = -1

        while True:
            now = time.perf_counter()
            elapsed = now - t0
            phase = "settle" if elapsed < settle_s else "collect"

            if elapsed >= settle_s + collect_s:
                break

            f = feature_provider()

            if f is not None and f.ok and f.seq != last_seq:
                last_seq = f.seq

                if phase == "collect" and f.ear > 0.12:       # Skip frames mid-blink
                    samples.append(feature_vector(f))
                    ears.append(f.ear)

            progress = 0.0 if phase == "settle" else min(1.0, (elapsed - settle_s) / collect_s)

            if not show(f"Point {i + 1}/{len(targets)}   samples: {len(samples)}", target, progress):
                cv2.destroyWindow(window_name)

                return None

        X_by_point.append(np.array(samples) if samples else np.zeros((0, FEATURE_DIM)))

    X = np.vstack([x for x in X_by_point if len(x)])
    Y = np.vstack([np.repeat([[tx, ty]], len(x), axis = 0) for x, (tx, ty) in zip(X_by_point, targets) if len(x)])

    if len(X) < 5 * len(targets):
        show("Too few samples: is the camera seeing your face? Press any key.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        return None

    model = fit_model(X, Y, screen_w, screen_h, ear_baseline = float(np.median(ears)) if ears else 0.0)
    model.point_errors_px = evaluate_per_point(model, X_by_point, targets)

    # Results screen: target circles with the mean prediction drawn on top ----
    def draw_results(c):
        for X_p, (tx, ty), err in zip(X_by_point, targets, model.point_errors_px):
            cv2.circle(c, (int(tx), int(ty)), 18, (90, 90, 90), 2)

            if len(X_p):
                pred = np.array([model.predict(v) for v in X_p])
                px, py = pred.mean(axis = 0)
                cv2.line(c, (int(tx), int(ty)), (int(px), int(py)), (0, 140, 255), 2)
                cv2.circle(c, (int(px), int(py)), 6, (0, 220, 0), -1)
                cv2.putText(c, f"{err:.0f}px", (int(tx) + 22, int(ty) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    t_end = time.perf_counter() + 4.0

    while time.perf_counter() < t_end:
        if not show(f"Done. RMSE {model.rmse_px:.0f} px, EAR baseline {model.ear_baseline:.2f}. "
                    f"Green = where the model thinks you looked.", extra = draw_results):
            break

    cv2.destroyWindow(window_name)

    return model


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    W, H = 1920, 1080
    pts = calibration_points()
    assert len(pts) == 13 and all(0 < x < 1 and 0 < y < 1 for x, y in pts)

    # Synthetic subject: screen x depends linearly on iris hx, y on vy, plus noise.
    X_by_point, targets = [], []

    for fx, fy in pts:
        tx, ty = fx * W, fy * H
        rows = []

        for _ in range(25):
            f = FaceFeatures(ok = True)
            f.hx_l = f.hx_r = (fx - 0.5) * 0.4 + rng.normal(0, 0.005)
            f.vy_l = f.vy_r = (fy - 0.5) * 0.25 + rng.normal(0, 0.005)
            f.yaw, f.pitch, f.tz = rng.normal(0, 0.01), rng.normal(0, 0.01), -40 + rng.normal(0, 0.2)
            rows.append(feature_vector(f))

        X_by_point.append(np.array(rows))
        targets.append((tx, ty))

    X = np.vstack(X_by_point)
    Y = np.vstack([np.repeat([[tx, ty]], 25, axis = 0) for tx, ty in targets])

    model = fit_model(X, Y, W, H, ear_baseline = 0.3)
    errs = evaluate_per_point(model, X_by_point, targets)
    assert model.rmse_px < 40, model.rmse_px
    assert max(errs) < 60, errs

    # Round trip through JSON gives identical predictions.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "calibration.json"
        model.save(p)
        back = CalibrationModel.load(p)
        assert back is not None and back.ear_baseline == 0.3
        v = X[0]
        assert np.allclose(model.predict(v), back.predict(v))

    assert CalibrationModel.load("/nonexistent/calibration.json") is None
    print(f"calibration: OK (synthetic RMSE {model.rmse_px:.1f} px)")


if __name__ == "__main__":
    _self_test()
