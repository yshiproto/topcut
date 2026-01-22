from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


def _largest_contour(contours: list[np.ndarray]) -> np.ndarray:
    if not contours:
        raise ValueError("No contours found for border detection")
    return max(contours, key=cv2.contourArea)


def detect_border(image: np.ndarray) -> Rect:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = _largest_contour(contours)
    x, y, width, height = cv2.boundingRect(contour)
    return Rect(x=x, y=y, width=width, height=height)
