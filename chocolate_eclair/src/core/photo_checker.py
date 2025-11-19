"""Image quality assessment helpers for the Eclair project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


class PhotoCheckerError(RuntimeError):
    """Raised when the photo quality analysis fails."""


@dataclass(frozen=True)
class PhotoMetrics:
    blur: float
    brightness: float
    resolution: str
    width: int
    height: int

    def as_dict(self) -> Dict[str, float | str | int]:
        return {
            "blur": self.blur,
            "brightness": self.brightness,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
        }


def _load_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise PhotoCheckerError(f"File not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise PhotoCheckerError(f"Unable to read image: {image_path}")
    return image


BLUR_THRESHOLD = 100.0
BRIGHTNESS_RANGE = (40.0, 200.0)
# 1920x1080 ~= 2.07 MP; allow anything at or above that resolution.
MIN_RESOLUTION_PIXELS = 2_000_000


def _calculate_blur(gray_image: np.ndarray) -> float:
    # Laplacian variance is a simple focus metric; higher values mean sharper images.
    variance = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    return variance


def _calculate_brightness(gray_image: np.ndarray) -> float:
    # Average pixel intensity on a 0-255 scale provides a rough brightness estimate.
    return float(np.mean(gray_image))


def _format_resolution(width: int, height: int) -> str:
    megapixels = (width * height) / 1_000_000
    return f"{width}x{height} ({megapixels:.2f} MP)"


def analyze_photo(image_path: Path | str) -> Dict[str, float | str | int | List[str]]:
    """Return basic quality metrics for the provided image."""

    path = Path(image_path)
    image = _load_image(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = _calculate_blur(gray)
    brightness = _calculate_brightness(gray)
    height, width = gray.shape
    metrics = PhotoMetrics(
        blur=blur,
        brightness=brightness,
        resolution=_format_resolution(width, height),
        width=width,
        height=height,
    )
    result = metrics.as_dict()

    issues: List[str] = []
    # if blur < BLUR_THRESHOLD:
    #     issues.append(f"Blur score low ({blur:.2f} < {BLUR_THRESHOLD:.0f})")

    min_brightness, max_brightness = BRIGHTNESS_RANGE
    # if brightness < min_brightness or brightness > max_brightness:
    #     issues.append(
    #         f"Brightness outside recommended range ({brightness:.2f} not in {int(min_brightness)}-{int(max_brightness)})"
    #     )

    # if (width * height) < MIN_RESOLUTION_PIXELS:
    #     issues.append("Resolution below 2 MP (minimum 1920x1080)")

    status = "OK" if not issues else "Needs attention"
    result["status"] = status
    result["issues"] = issues
    return result
