from __future__ import annotations

from dataclasses import dataclass

from topcut.detection.border import Rect
from topcut.detection.content import ContentRect


@dataclass(frozen=True)
class MarginMetrics:
    left: float
    right: float
    top: float
    bottom: float
    horizontal_centering_pct: float
    vertical_centering_pct: float


def _centering_percentage(near: float, far: float) -> float:
    total = near + far
    if total <= 0:
        return 0.0
    return (near / total) * 100.0


def compute_margins(border: Rect, content: ContentRect) -> MarginMetrics:
    left = content.x - border.x
    right = (border.x + border.width) - (content.x + content.width)
    top = content.y - border.y
    bottom = (border.y + border.height) - (content.y + content.height)
    horizontal_centering_pct = _centering_percentage(left, right)
    vertical_centering_pct = _centering_percentage(top, bottom)
    return MarginMetrics(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        horizontal_centering_pct=horizontal_centering_pct,
        vertical_centering_pct=vertical_centering_pct,
    )
