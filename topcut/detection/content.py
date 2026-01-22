from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from topcut.detection.border import Rect


@dataclass(frozen=True)
class ContentRect:
    x: int
    y: int
    width: int
    height: int


def _largest_contour(contours: list[np.ndarray]) -> np.ndarray:
    if not contours:
        raise ValueError("No contours found for content detection")
    return max(contours, key=cv2.contourArea)


def detect_content(image: np.ndarray, border: Rect) -> ContentRect:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = gray[border.y : border.y + border.height, border.x : border.x + border.width]
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = _largest_contour(contours)
    x, y, width, height = cv2.boundingRect(contour)
    return ContentRect(
        x=x + border.x,
        y=y + border.y,
        width=width,
        height=height,
    )
