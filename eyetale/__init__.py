"""
EyeTale: play Undertale with your eyes.

Pipeline:  camera -> face_tracker -> calibration model -> gaze_estimator
           -> (gestures, screen_reader, mapper) -> input_driver -> game

Run modes live in eyetale.app:  python -m eyetale.app --help
"""

__version__ = "0.1.0"
