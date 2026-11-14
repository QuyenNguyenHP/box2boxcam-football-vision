import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from box2boxcam.data.luna_crop_pilot import (
    budget_bound,
    crop_geometry,
    restore_bbox,
    select_cases,
)


class LunaCropPilotTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/luna_crop_pilot.json").read_text())

    def test_mapping_accounts_for_border_and_zoom(self):
        g = crop_geometry([500, 800, 520, 820], 1080, 1920, self.config)
        b = g["border"]
        x, y, _, _ = g["crop_box"]
        local = [
            round(b + (500 - x) * g["scale_x"]),
            round(b + (800 - y) * g["scale_y"]),
            round(b + (520 - x) * g["scale_x"]),
            round(b + (820 - y) * g["scale_y"]),
        ]
        self.assertEqual(restore_bbox(local, g), [500, 800, 520, 820])

    def test_roi_stays_inside_image_and_rejects_margin_boxes(self):
        g = crop_geometry([0, 0, 5, 5], 720, 1280, self.config)
        self.assertEqual(g["crop_box"][:2], [0, 0])
        with self.assertRaises(ValueError):
            restore_bbox([0, 0, 4, 4], g)

    def test_budget_uses_cache_write_price_and_full_output_ceiling(self):
        self.assertAlmostEqual(budget_bound(200000, 40, self.config), 0.34018)
        self.assertGreater(budget_bound(3000000, 40, self.config), 0.50)

    def test_selection_covers_six_videos_without_duplicates(self):
        rows = [
            {
                "video_id": str(v),
                "item_id": f"{v}_{i}",
                "confidence": i / 100,
                "ball_in_play_visible": True,
                "bbox_current_pixels": [0, 0, 1, 1],
            }
            for v in range(6)
            for i in range(100)
        ]
        selected = select_cases(rows, 20)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len({r["item_id"] for r in selected}), 20)
        self.assertEqual(len({r["video_id"] for r in selected}), 6)


if __name__ == "__main__":
    unittest.main()
