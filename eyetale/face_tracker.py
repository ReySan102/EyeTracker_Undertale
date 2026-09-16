"""
Face and iris tracking with MediaPipe Face Landmarker.

One camera frame in, one `FaceFeatures` out. The features are the raw
material for two consumers:

  * the calibration model turns them into a screen coordinate (gaze)
  * the gesture detector watches the eye aspect ratio (blinks)

MediaPipe gives 478 landmarks per face. The ones this module cares about:

    right eye  (subject's right)   corners 33 / 133, lids 159 (top) 145 (bottom)
    left eye                       corners 263 / 362, lids 386 (top) 374 (bottom)
    right iris centre 468          left iris centre 473

Feature design: the iris centre is expressed relative to the eye's corner
midpoint and divided by the eye width. That makes it independent of where
the face is in the frame and of eyelid position. Head pose (rotation and
translation from MediaPipe's transformation matrix) is added so that the
regression can compensate for head movement.

The pure helpers (eye_aspect_ratio, iris_offset, feature_vector) have no
MediaPipe dependency and are unit-tested here.

Run:  python -m eyetale.face_tracker
"""
from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Landmark indices ------------------------------------------------------------
R_EYE_OUTER, R_EYE_INNER, R_EYE_TOP, R_EYE_BOTTOM = 33, 133, 159, 145
L_EYE_OUTER, L_EYE_INNER, L_EYE_TOP, L_EYE_BOTTOM = 263, 362, 386, 374
R_IRIS, L_IRIS = 468, 473
# Six-point sets for the eye aspect ratio (Soukupova & Cech 2016):
# p1 corner, p2/p3 upper lid, p4 other corner, p5/p6 lower lid.
R_EAR_IDX = (33, 160, 158, 133, 153, 144)
L_EAR_IDX = (362, 385, 387, 263, 373, 380)

FEATURE_DIM = 21


@dataclass
class FaceFeatures:
    ok: bool = False
    t: float = 0.0                 # Capture time, perf_counter seconds
    seq: int = 0
    # Iris offsets, normalised by eye width: negative = toward the subject's right / up
    hx_l: float = 0.0
    vy_l: float = 0.0
    hx_r: float = 0.0
    vy_r: float = 0.0
    ear_l: float = 0.0
    ear_r: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    blink_l: float = 0.0           # MediaPipe blendshape scores, 0..1 (informational)
    blink_r: float = 0.0
    frame_w: int = 0
    frame_h: int = 0
    landmarks_px: np.ndarray | None = field(default = None, repr = False)   # (478, 2) for drawing

    @property
    def ear(self) -> float:
        return min(self.ear_l, self.ear_r)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def eye_aspect_ratio(p: np.ndarray) -> float:
    """p: (6, 2) points ordered p1..p6. Open eye ~0.25-0.35, closed ~0.05-0.15."""
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])

    if h < 1e-6:
        return 0.0

    return float((v1 + v2) / (2.0 * h))


def iris_offset(iris: np.ndarray, corner_a: np.ndarray, corner_b: np.ndarray) -> tuple[float, float]:
    """Iris centre relative to the corner midpoint, in units of eye width."""
    mid = (corner_a + corner_b) / 2.0
    width = float(np.linalg.norm(corner_b - corner_a))

    if width < 1e-6:
        return 0.0, 0.0

    d = (iris - mid) / width

    return float(d[0]), float(d[1])


def head_pose_from_matrix(m: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """
    (yaw, pitch, roll, tx, ty, tz) from MediaPipe's 4x4 facial transformation matrix.
    Any smooth, consistent decomposition is fine: the regression only needs the
    numbers to vary monotonically with head movement.
    """
    r = m[:3, :3]
    yaw = math.atan2(r[0, 2], r[2, 2])
    pitch = math.atan2(-r[1, 2], math.sqrt(r[0, 2] ** 2 + r[2, 2] ** 2))
    roll = math.atan2(r[1, 0], r[1, 1])

    return yaw, pitch, roll, float(m[0, 3]), float(m[1, 3]), float(m[2, 3])


def feature_vector(f: FaceFeatures) -> np.ndarray:
    """
    Expand the raw features into the regression input (FEATURE_DIM values).
    Squares and cross terms let a linear ridge fit model the mild curvature
    of the eye-to-screen mapping; head terms compensate for head movement.
    """
    hl, vl, hr, vr = f.hx_l, f.vy_l, f.hx_r, f.vy_r

    return np.array([
        1.0,
        hl, vl, hr, vr,
        hl * hl, vl * vl, hr * hr, vr * vr,
        hl * vl, hr * vr,
        f.yaw, f.pitch, f.roll,
        f.tx, f.ty, f.tz,
        hl * f.yaw, hr * f.yaw, vl * f.pitch, vr * f.pitch,
    ], dtype = np.float64)


def features_from_landmarks(pts: np.ndarray, matrix: np.ndarray | None = None) -> FaceFeatures:
    """Build FaceFeatures from a (478, 2) pixel landmark array. Pure; testable."""
    f = FaceFeatures(ok = True)
    f.hx_r, f.vy_r = iris_offset(pts[R_IRIS], pts[R_EYE_OUTER], pts[R_EYE_INNER])
    f.hx_l, f.vy_l = iris_offset(pts[L_IRIS], pts[L_EYE_INNER], pts[L_EYE_OUTER])
    f.ear_r = eye_aspect_ratio(pts[list(R_EAR_IDX)])
    f.ear_l = eye_aspect_ratio(pts[list(L_EAR_IDX)])

    if matrix is not None:
        f.yaw, f.pitch, f.roll, f.tx, f.ty, f.tz = head_pose_from_matrix(matrix)
    else:
        # Fallback head features from the landmarks themselves: face centre and size.
        nose, chin, brow = pts[1], pts[152], pts[10]
        f.tx, f.ty = float(nose[0]), float(nose[1])
        f.tz = float(np.linalg.norm(chin - brow))

    f.landmarks_px = pts

    return f


def ensure_model(path: str | Path) -> Path:
    """Download the Face Landmarker model on first use (about 3.7 MB)."""
    p = Path(path)

    if p.exists() and p.stat().st_size > 100_000:
        return p

    p.parent.mkdir(parents = True, exist_ok = True)
    print(f"[face_tracker] downloading model to {p} ...")
    urllib.request.urlretrieve(MODEL_URL, p)

    return p


# ---------------------------------------------------------------------------
# MediaPipe wrapper
# ---------------------------------------------------------------------------

class FaceTracker:
    """Wraps MediaPipe FaceLandmarker in VIDEO mode. Call `process` with frames in capture order; timestamps must increase."""

    def __init__(self, model_path: str | Path = "models/face_landmarker.task") -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model = ensure_model(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options = mp_python.BaseOptions(model_asset_path = str(model)),
            running_mode = vision.RunningMode.VIDEO,
            num_faces = 1,
            output_face_blendshapes = True,
            output_facial_transformation_matrixes = True,
            min_face_detection_confidence = 0.5,
            min_face_presence_confidence = 0.5,
            min_tracking_confidence = 0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self.process_ms: float = 0.0     # Last inference time, for the HUD

    def process(self, frame_bgr: np.ndarray, t: float, seq: int = 0) -> FaceFeatures:
        import cv2

        ts_ms = int(t * 1000)

        if ts_ms <= self._last_ts_ms:          # MediaPipe requires strictly increasing timestamps
            ts_ms = self._last_ts_ms + 1

        self._last_ts_ms = ts_ms

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format = self._mp.ImageFormat.SRGB, data = np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts_ms)
        self.process_ms = (time.perf_counter() - t0) * 1000.0

        h, w = frame_bgr.shape[:2]

        if not result.face_landmarks:
            return FaceFeatures(ok = False, t = t, seq = seq, frame_w = w, frame_h = h)

        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype = np.float64)
        matrix = None

        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0], dtype = np.float64)

        f = features_from_landmarks(pts, matrix)
        f.t, f.seq, f.frame_w, f.frame_h = t, seq, w, h

        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            f.blink_l = float(scores.get("eyeBlinkLeft", 0.0))
            f.blink_r = float(scores.get("eyeBlinkRight", 0.0))

        return f

    def close(self) -> None:
        self._landmarker.close()


def draw_features(frame_bgr: np.ndarray, f: FaceFeatures) -> np.ndarray:
    """Overlay eye landmarks and EAR values on a copy of the frame (HUD preview)."""
    import cv2

    out = frame_bgr.copy()

    if not f.ok or f.landmarks_px is None:
        cv2.putText(out, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return out

    pts = f.landmarks_px

    for idx in R_EAR_IDX + L_EAR_IDX:
        cv2.circle(out, (int(pts[idx][0]), int(pts[idx][1])), 2, (0, 255, 0), -1)

    for idx in (R_IRIS, L_IRIS):
        cv2.circle(out, (int(pts[idx][0]), int(pts[idx][1])), 3, (255, 0, 255), -1)

    cv2.putText(out, f"EAR {f.ear_l:.2f}/{f.ear_r:.2f}  yaw {math.degrees(f.yaw):+.0f} pitch {math.degrees(f.pitch):+.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return out


# ---------------------------------------------------------------------------
# Self-test (pure helpers only; MediaPipe is exercised by the calibrate command)
# ---------------------------------------------------------------------------

def _synthetic_landmarks(iris_shift = (0.0, 0.0), closed: bool = False) -> np.ndarray:
    """A fake 478-point array with plausible eye geometry."""
    pts = np.zeros((478, 2))
    pts[1] = (320, 260)      # Nose
    pts[10] = (320, 120)     # Brow
    pts[152] = (320, 400)    # Chin
    lid = 2.0 if closed else 10.0

    # Right eye centred at (260, 220), width 40; left eye at (380, 220)
    for cx, outer, inner, top, bottom, iris, ear_idx in (
        (260, R_EYE_OUTER, R_EYE_INNER, R_EYE_TOP, R_EYE_BOTTOM, R_IRIS, R_EAR_IDX),
        (380, L_EYE_INNER, L_EYE_OUTER, L_EYE_TOP, L_EYE_BOTTOM, L_IRIS, L_EAR_IDX),
    ):
        pts[outer] = (cx - 20, 220)
        pts[inner] = (cx + 20, 220)
        pts[top] = (cx, 220 - lid)
        pts[bottom] = (cx, 220 + lid)
        pts[iris] = (cx + iris_shift[0] * 40, 220 + iris_shift[1] * 40)
        p1, p2, p3, p4, p5, p6 = ear_idx
        pts[p1], pts[p4] = pts[outer], pts[inner]
        pts[p2], pts[p3] = (cx - 7, 220 - lid), (cx + 7, 220 - lid)
        pts[p6], pts[p5] = (cx - 7, 220 + lid), (cx + 7, 220 + lid)

    return pts


def _self_test() -> None:
    open_eye = _synthetic_landmarks()
    closed_eye = _synthetic_landmarks(closed = True)
    f_open = features_from_landmarks(open_eye)
    f_closed = features_from_landmarks(closed_eye)
    assert f_open.ok and f_open.ear > 0.4 and f_closed.ear < 0.15, (f_open.ear, f_closed.ear)
    assert abs(f_open.hx_l) < 1e-9 and abs(f_open.hx_r) < 1e-9       # Iris centred

    f_right = features_from_landmarks(_synthetic_landmarks(iris_shift = (0.2, -0.1)))
    assert abs(f_right.hx_r - 0.2) < 1e-9 and abs(f_right.vy_r + 0.1) < 1e-9

    vec = feature_vector(f_right)
    assert vec.shape == (FEATURE_DIM,) and vec[0] == 1.0

    identity = np.eye(4)
    identity[:3, 3] = (1.0, 2.0, -30.0)
    yaw, pitch, roll, tx, ty, tz = head_pose_from_matrix(identity)
    assert abs(yaw) < 1e-9 and abs(pitch) < 1e-9 and abs(roll) < 1e-9 and tz == -30.0

    # A pure yaw rotation must show up in the yaw channel.
    a = math.radians(20)
    ry = np.array([[math.cos(a), 0, math.sin(a), 0], [0, 1, 0, 0], [-math.sin(a), 0, math.cos(a), 0], [0, 0, 0, 1]])
    yaw, pitch, roll, *_ = head_pose_from_matrix(ry)
    assert abs(yaw - a) < 1e-9 and abs(pitch) < 1e-9 and abs(roll) < 1e-9
    print("face_tracker: OK (pure helpers)")


if __name__ == "__main__":
    _self_test()
