import unittest

from pitchlens.data.build_mobile_dataset import (
    append_reviewed_train, check_groups, select_luna_rows, select_tv_rows,
)


class MobileSelectionTests(unittest.TestCase):
    def test_missing_response_is_not_a_negative(self):
        row = {"split": "train", "status": "negative", "luna_status": "réponse Luna manquante"}
        selected, omitted = select_luna_rows([row])
        self.assertEqual(selected, [])
        self.assertEqual(omitted, [row])

    def test_human_positive_survives_missing_original_response(self):
        row = {"split": "val", "status": "positive", "luna_status": "réponse Luna manquante"}
        self.assertEqual(select_luna_rows([row]), ([row], []))

    def test_test_is_never_selected(self):
        self.assertEqual(select_luna_rows([{"split": "test"}]), ([], []))

    def test_match_cannot_cross_train_validation(self):
        rows = [{"source_dataset": "luna", "match_id": "one", "split": split} for split in ("train", "val")]
        with self.assertRaises(ValueError):
            check_groups(rows)

    def test_multiple_frames_in_same_match_are_valid(self):
        rows = [{"source_dataset": "luna", "match_id": "one", "split": "train"}] * 2
        check_groups(rows)

    def test_tv_does_not_include_reserved_splits_or_exclusions(self):
        rows = [dict(split=split, sequence=split, output_image="image.jpg", decision_applied="keep")
                for split in ("train", "val", "test")]
        rows.append(dict(split="train", sequence="exclude", output_image="", decision_applied="exclude"))
        self.assertEqual(select_tv_rows(rows), [rows[0]])

    def test_tv_sequence_overlap_is_rejected(self):
        rows = [dict(split=split, sequence="same", output_image="image.jpg", decision_applied="keep")
                for split in ("train", "test")]
        with self.assertRaises(ValueError):
            select_tv_rows(rows)

    def test_enrichment_is_train_only_and_preserves_validation(self):
        validation = {"source_dataset": "luna", "match_id": "one", "split": "val"}
        rows = [validation.copy()]
        candidate = dict(item_id="frame", source_frame=0, status="positive",
                         label_source="submitted_human", match_id="", video_id="clip",
                         source_video_sha256="abc", split="unassigned")
        append_reviewed_train(rows, "shorts", [candidate, {"split": "excluded"},
                                              {"split": "not_selected"}])
        self.assertEqual(rows[0], validation)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["split"], "train")
        self.assertFalse(rows[1]["match_id_verified"])
        self.assertEqual(rows[1]["source_layout"], "unsplit")

    def test_enrichment_never_imports_reserved_test(self):
        with self.assertRaises(ValueError):
            append_reviewed_train([], "shorts", [{"split": "test", "status": "positive"}])

    def test_enrichment_rejects_invalid_source_path(self):
        with self.assertRaises(ValueError):
            append_reviewed_train([], "../test", [])

    def test_added_frozen_hashes_are_preserved_for_preflight(self):
        candidate = dict(item_id="frame", source_frame=0, status="negative",
                         label_source="user_confirmed_negative_video", match_id="session",
                         video_id="clip", source_video_sha256="source", split="unassigned",
                         image_sha256="image", label_sha256="label")
        rows = []
        append_reviewed_train(rows, "hard_negatives", [candidate])
        self.assertEqual(rows[0]["image_sha256"], "image")
        self.assertEqual(rows[0]["label_sha256"], "label")
