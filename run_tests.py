"""
Run every module's self-test block in build order.

    python run_tests.py

Modules that need hardware (camera) or Windows report SKIP instead of failing.
"""
from __future__ import annotations

import importlib
import sys
import traceback

MODULES = [
    "eyetale.config",
    "eyetale.filters",
    "eyetale.geometry",
    "eyetale.input_driver",
    "eyetale.winapi",
    "eyetale.camera",
    "eyetale.face_tracker",
    "eyetale.calibration",
    "eyetale.gaze_estimator",
    "eyetale.gestures",
    "eyetale.game_window",
    "eyetale.screen_reader",
    "eyetale.mapper",
    "eyetale.hud",
]


def main() -> int:
    failed = []

    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod._self_test()
        except Exception:
            failed.append(name)
            print(f"{name}: FAIL")
            traceback.print_exc()

    print("-" * 40)
    print("all self-tests passed" if not failed else f"failed: {', '.join(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
