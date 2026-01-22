from __future__ import annotations

import argparse
import json
from pathlib import Path

from topcut.detection.border import detect_border
from topcut.detection.content import detect_content
from topcut.image_io import load_image, normalize_image, resize_image
from topcut.metrics import compute_margins
from topcut.visualize import draw_overlay, save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trading card centering.")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument(
        "output",
        help="Path to save debug overlay image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.input)
    loaded = resize_image(image)
    _ = normalize_image(loaded.resized)

    border = detect_border(loaded.resized)
    content = detect_content(loaded.resized, border)
    metrics = compute_margins(border, content)

    overlay = draw_overlay(loaded.resized, border, content)
    save_overlay(Path(args.output), overlay)

    summary = {
        "border": {
            "x": border.x,
            "y": border.y,
            "width": border.width,
            "height": border.height,
        },
        "content": {
            "x": content.x,
            "y": content.y,
            "width": content.width,
            "height": content.height,
        },
        "margins": {
            "left": metrics.left,
            "right": metrics.right,
            "top": metrics.top,
            "bottom": metrics.bottom,
        },
        "centering": {
            "horizontal_pct": metrics.horizontal_centering_pct,
            "vertical_pct": metrics.vertical_centering_pct,
        },
        "scale": loaded.scale,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
