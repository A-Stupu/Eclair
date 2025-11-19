"""Capture helper: take photos with a webcam, run quick analysis and
coordinate turntable movement via the Arduino controller.

This module keeps responsibilities small so the GUI can call into it
from a background thread.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2

from chocolate_eclair.src.core.photo_checker import analyze_photo

LOG = logging.getLogger(__name__)


def _preferred_backend() -> int:
    if sys.platform.startswith("win"):
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    if sys.platform.startswith("linux"):
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    if sys.platform == "darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    return cv2.CAP_ANY


def _open_with_backend(index: int, backend: int) -> Optional[cv2.VideoCapture]:
    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception:
        return None
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None
    return cap


def _open_camera(index: int = 0) -> cv2.VideoCapture:
    backend = _preferred_backend()
    cap = _open_with_backend(index, backend)
    if not cap and backend != cv2.CAP_ANY:
        cap = _open_with_backend(index, cv2.CAP_ANY)
    if not cap:
        raise RuntimeError(f"Unable to open camera index {index}")
    return cap


def capture_image(
    save_path: Path,
    camera_index: int = 0,
    capture: Optional[cv2.VideoCapture] = None,
) -> Path:
    """Capture a single frame from the camera and write to save_path.

    Returns the path to the saved file.
    """
    if capture is None:
        cap = _open_camera(camera_index)
        release_needed = True
    else:
        cap = capture
        release_needed = False

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), frame)
        LOG.info("Captured image: %s", save_path)
        return save_path
    finally:
        if release_needed:
            cap.release()


def capture_and_analyze(
    save_path: Path,
    camera_index: int = 0,
    arduino=None,
    advance_degrees: float = 10.0,
    capture: Optional[cv2.VideoCapture] = None,
) -> Dict[str, object]:
    """Capture an image, analyze it, and optionally advance the turntable.

    Behavior:
      - capture image to save_path
      - if arduino provided, rotate forward by advance_degrees
      - analyze the captured image
      - if analysis reports not OK and arduino was used, rotate back by advance_degrees

    Returns a dict with keys: path, metrics, ok (bool)
    """
    path = capture_image(save_path, camera_index=camera_index, capture=capture)
    used_arduino = arduino is not None

    # move forward immediately after capture to prepare next position
    if used_arduino:
        try:
            arduino.rotate_degrees(advance_degrees)
        except Exception:
            LOG.exception("Failed to advance turntable after capture")

    metrics = analyze_photo(path)
    status = metrics.get("status", "Unknown")
    ok = str(status).upper() == "OK"

    if not ok and used_arduino:
        # go back to previous position so user can retake
        try:
            arduino.rotate_degrees(-advance_degrees)
        except Exception:
            LOG.exception("Failed to move turntable back after bad capture")

    return {"path": path, "metrics": metrics, "ok": ok}
