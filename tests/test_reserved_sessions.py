"""A reserved match stays out of fitting even when a clip is renamed."""

import json
import tempfile
import unittest
from pathlib import Path

from box2boxcam.data.reserved_sessions import assert_no_reserved_training


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        self.registry = self.root / "configs/reserved_sessions.json"
        self.registry.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sessions": [
                        {
                            "match_id": "held_out",
                            "forbidden_splits": ["train"],
                            "videos": [
                                {
                                    "source_video_sha256": "abc",
                                    "source_video": "clip.mov",
                                }
                            ],
                        }
                    ],
                }
            )
        )

    def test_match_hash_or_source_path_prevents_training(self):
        for identity in [
            {"match_id": "held_out"},
            {"source_video_sha256": "abc"},
            {"source_video": "clip.mov"},
        ]:
            with (
                self.subTest(identity=identity),
                self.assertRaisesRegex(ValueError, "Reserved field session"),
            ):
                assert_no_reserved_training(self.root, [{"split": "train", **identity}])

    def test_validation_and_unrelated_training_are_allowed(self):
        assert_no_reserved_training(
            self.root,
            [
                {"split": "val", "match_id": "held_out"},
                {"split": "train", "match_id": "another_match"},
            ],
        )

    def test_corrupt_policy_cannot_silently_disable_protection(self):
        self.registry.write_text('{"schema_version": 2, "sessions": []}')
        with self.assertRaisesRegex(ValueError, "Invalid"):
            assert_no_reserved_training(self.root, [])


if __name__ == "__main__":
    unittest.main()
