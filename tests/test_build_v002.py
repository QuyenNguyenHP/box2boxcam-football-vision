import unittest

from box2boxcam.data.build_v002 import resolve_label, validate_label


class BuildPolicyTests(unittest.TestCase):
    def test_original_multi_boxes_and_explicit_decisions(self):
        original = "0 0.3 0.4 0.1 0.1\n0 0.7 0.6 0.1 0.1\n"
        self.assertEqual(resolve_label(original, None), ("default_keep", original))
        self.assertEqual(validate_label(original), 2)
        self.assertEqual(resolve_label(original, {"decision": "negative"}), ("negative", ""))
        action, label = resolve_label(original, {
            "decision": "bbox", "x_center": "0.5", "y_center": "0.5",
            "width": "0.2", "height": "0.1",
        })
        self.assertEqual(action, "bbox")
        self.assertEqual(validate_label(label), 1)

    def test_invalid_unreviewed_label_blocks_build(self):
        with self.assertRaises(ValueError):
            resolve_label("0 0 0 0 0", None)
        self.assertEqual(resolve_label("0 0 0 0 0", {"decision": "exclude"}), ("exclude", ""))
