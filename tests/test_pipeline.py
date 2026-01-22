import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from topcut.detection.border import detect_border
from topcut.detection.content import detect_content
from topcut.metrics import compute_margins
from topcut.visualize import draw_overlay, save_overlay


class PipelineTests(unittest.TestCase):
    def _make_synthetic_image(self) -> np.ndarray:
        image = np.zeros((700, 500, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 40), (450, 640), (50, 50, 50), -1)
        cv2.rectangle(image, (100, 120), (400, 520), (200, 200, 200), -1)
        return image

    def test_detection_and_metrics(self) -> None:
        image = self._make_synthetic_image()
        border = detect_border(image)
        content = detect_content(image, border)
        metrics = compute_margins(border, content)

        self.assertLessEqual(abs(border.x - 50), 2)
        self.assertLessEqual(abs(border.y - 40), 2)
        self.assertLessEqual(abs(border.width - 401), 2)
        self.assertLessEqual(abs(border.height - 601), 2)

        self.assertLessEqual(abs(content.x - 100), 2)
        self.assertLessEqual(abs(content.y - 120), 2)
        self.assertLessEqual(abs(content.width - 301), 2)
        self.assertLessEqual(abs(content.height - 401), 2)

        self.assertAlmostEqual(metrics.left, 50, delta=2)
        self.assertAlmostEqual(metrics.right, 50, delta=2)
        self.assertAlmostEqual(metrics.top, 80, delta=2)
        self.assertAlmostEqual(metrics.bottom, 80, delta=2)
        self.assertAlmostEqual(metrics.horizontal_centering_pct, 50.0, delta=1.0)
        self.assertAlmostEqual(metrics.vertical_centering_pct, 50.0, delta=1.0)

    def test_overlay_saved(self) -> None:
        image = self._make_synthetic_image()
        border = detect_border(image)
        content = detect_content(image, border)
        overlay = draw_overlay(image, border, content)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "overlay.png"
            save_overlay(output_path, overlay)
            self.assertTrue(output_path.exists())
            saved = cv2.imread(str(output_path))
            self.assertIsNotNone(saved)


if __name__ == "__main__":
    unittest.main()
