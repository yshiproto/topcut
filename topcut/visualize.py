from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from topcut.detection.border import Rect
from topcut.detection.content import ContentRect


def draw_overlay(
    image: np.ndarray, border: Rect, content: ContentRect
) -> np.ndarray:
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (border.x, border.y),
        (border.x + border.width, border.y + border.height),
        (0, 255, 0),
        3,
    )
    cv2.rectangle(
        overlay,
        (content.x, content.y),
        (content.x + content.width, content.y + content.height),
        (0, 0, 255),
        3,
    )
    return overlay


def save_overlay(path: str | Path, overlay: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise IOError(f"Unable to write overlay image to {output_path}")
