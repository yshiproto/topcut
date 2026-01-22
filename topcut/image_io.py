from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class LoadedImage:
    original: np.ndarray
    resized: np.ndarray
    scale: float


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {path}")
    return image


def resize_image(image: np.ndarray, max_dim: int = 1200) -> LoadedImage:
    height, width = image.shape[:2]
    scale = 1.0
    if max(height, width) > max_dim:
        scale = max_dim / float(max(height, width))
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
    return LoadedImage(original=image, resized=resized, scale=scale)


def normalize_image(image: np.ndarray) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    return normalized
