"""Regression checks for preserving the actual COCO sports-ball classifier."""

import copy
import io
import unittest
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from pitchlens.training.sports_ball_transfer import (
    SportsBallTransferTrainer,
    retain_sports_ball,
    verify_dense_predictions,
)

WEIGHTS = Path(__file__).resolve().parents[1] / "models/checkpoints/yolo26n.pt"


@unittest.skipUnless(WEIGHTS.is_file(), "Local pretrained YOLO26n weights are required")
class SportsBallTransferTests(unittest.TestCase):
    weights = WEIGHTS
    classifier_width = 80

    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, cls.original_threads)
        cls.source = YOLO(str(cls.weights)).model.float().eval()

    def test_both_branches_copy_ball_row_and_preserve_every_other_tensor(self):
        original_state = {key: value.clone() for key, value in self.source.state_dict().items()}
        original_names = dict(self.source.names)
        original_yaml = copy.deepcopy(self.source.yaml)
        model, report = retain_sports_ball(self.source)
        self.assertIs(type(model), DetectionModel)
        self.assertEqual(report["source_class_id"], 32)
        self.assertEqual(report["target_class_id"], 0)
        self.assertTrue(report["all_weight_checks_passed"])
        self.assertEqual(model.names, {0: "sports ball"})
        self.assertEqual(model.yaml["nc"], 1)
        self.assertEqual(model.model[-1].nc, 1)
        self.assertEqual(model.model[-1].no, 4 * model.model[-1].reg_max + 1)
        self.assertEqual(report["classification_hidden_widths"], [self.classifier_width] * 6)
        mapped = set(report["mapped_output_tensors"])
        self.assertEqual(len(mapped), 12)
        for branch in ("cv3", "one2one_cv3"):
            for source_sequence, target_sequence in zip(
                getattr(self.source.model[-1], branch), getattr(model.model[-1], branch)
            ):
                self.assertEqual(target_sequence[-1].out_channels, 1)
                self.assertEqual(target_sequence[-1].in_channels, self.classifier_width)
                self.assertTrue(torch.equal(target_sequence[-1].weight, source_sequence[-1].weight[32:33]))
                self.assertTrue(torch.equal(target_sequence[-1].bias, source_sequence[-1].bias[32:33]))
        for key, target_value in model.state_dict().items():
            expected = original_state[key][32:33] if key in mapped else original_state[key]
            self.assertTrue(torch.equal(target_value, expected), key)
            self.assertTrue(torch.equal(self.source.state_dict()[key], original_state[key]), key)
            self.assertNotEqual(target_value.data_ptr(), self.source.state_dict()[key].data_ptr(), key)
        self.assertEqual(self.source.names, original_names)
        self.assertEqual(self.source.yaml, original_yaml)
        self.assertEqual(self.source.model[-1].nc, 80)

    def test_raw_and_decoded_predictions_preserve_original_ball(self):
        target, _ = retain_sports_ball(self.source)
        target.train()
        sample = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(73))
        report = verify_dense_predictions(self.source, target, sample)
        self.assertTrue(report["passed"])
        self.assertFalse(self.source.training)
        self.assertTrue(target.training)
        for branch in ("one2many", "one2one"):
            self.assertLess(report["branches"][branch]["class_logit_max_error"], 1e-4)
            self.assertEqual(report["branches"][branch]["raw_box_max_error"], 0.0)
            self.assertEqual(report["branches"][branch]["decoded_box_max_error"], 0.0)

    def test_missing_or_ambiguous_class_is_rejected(self):
        for names in (
            {**self.source.names, 32: "ball"},
            {**self.source.names, 31: "sports ball"},
        ):
            with self.subTest(names=names):
                source = copy.deepcopy(self.source)
                source.names = names
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    retain_sports_ball(source)

    def test_fused_source_is_rejected(self):
        source = copy.deepcopy(self.source)
        source.fuse(verbose=False)
        self.assertTrue(source.model[-1].cv3 is None or source.model[-1].one2one_cv3 is None)
        with self.assertRaisesRegex(ValueError, "unfused"):
            retain_sports_ball(source)

    def test_inconsistent_or_unsupported_structure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "standard"):
            retain_sports_ball(torch.nn.Linear(2, 1))
        source = copy.deepcopy(self.source)
        source.model[-1].nc -= 1
        with self.assertRaisesRegex(ValueError, "metadata"):
            retain_sports_ball(source)
        source = copy.deepcopy(self.source)
        source.model[-1].cv3 = torch.nn.ModuleList(list(source.model[-1].cv3)[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            retain_sports_ball(source)
        source = copy.deepcopy(self.source)
        source.model[-1].one2one_cv3[0][-1].bias = None
        with self.assertRaisesRegex(ValueError, "convolution"):
            retain_sports_ball(source)

    def test_standard_model_save_reload_keeps_graph_and_predictions(self):
        model, _ = retain_sports_ball(self.source)
        model.eval()
        buffer = io.BytesIO()
        torch.save({"model": model}, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)["model"].eval()
        self.assertIs(type(restored), DetectionModel)
        self.assertEqual(restored.names, {0: "sports ball"})
        self.assertEqual(restored.model[-1].no, 5)
        self.assertEqual([sequence[-1].in_channels for sequence in restored.model[-1].one2one_cv3], [self.classifier_width] * 3)
        sample = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(9))
        with torch.no_grad():
            before = model(sample)[1]
            after = restored(sample)[1]
        for branch in ("one2many", "one2one"):
            for key in ("scores", "boxes"):
                self.assertTrue(torch.equal(before[branch][key], after[branch][key]))

    def test_finite_detection_loss_and_backward_on_both_branches(self):
        model, _ = retain_sports_ball(self.source)
        model.args = get_cfg()
        model.train()
        batch = {
            "img": torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(8)),
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.zeros(2, 1),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.6, 0.4, 0.1, 0.1]]),
        }
        loss, components = model.loss(batch)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(all(torch.isfinite(value).all() for value in components.values()))
        loss.sum().backward()
        for branch in ("cv3", "one2one_cv3"):
            gradient = getattr(model.model[-1], branch)[0][-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_custom_trainer_uses_retained_classifier_without_shrinking(self):
        trainer = object.__new__(SportsBallTransferTrainer)
        trainer.data = {"nc": 1, "names": {0: "sports ball"}}
        model = trainer.get_model(cfg=self.source.yaml, weights=self.source, verbose=False)
        self.assertIs(type(model), DetectionModel)
        self.assertEqual([sequence[-1].in_channels for sequence in model.model[-1].cv3], [self.classifier_width] * 3)
        self.assertTrue(trainer.sports_ball_transfer_report["dense_prediction_check"]["passed"])
        self.assertEqual(self.source.model[-1].nc, 80)

    def test_custom_trainer_rejects_missing_weights_or_wrong_dataset(self):
        for data, weights in (
            ({"nc": 1, "names": {0: "sports ball"}}, None),
            ({"nc": 1, "names": {0: "ball"}}, self.source),
            ({"nc": 80, "names": self.source.names}, self.source),
        ):
            with self.subTest(data=data, has_weights=weights is not None):
                trainer = object.__new__(SportsBallTransferTrainer)
                trainer.data = data
                with self.assertRaisesRegex(ValueError, "one named class"):
                    trainer.get_model(weights=weights, verbose=False)

    def test_continuation_preserves_the_entire_one_class_graph(self):
        previous, _ = retain_sports_ball(self.source)
        trainer = object.__new__(SportsBallTransferTrainer)
        trainer.data = {"nc": 1, "names": {0: "sports ball"}}
        continued = trainer.get_model(cfg=previous.yaml, weights=previous, verbose=False)
        report = trainer.sports_ball_transfer_report
        self.assertEqual(report["source_class_id"], 0)
        self.assertEqual(report["source_classes"], 1)
        self.assertEqual(report["classification_hidden_widths"], [self.classifier_width] * 6)
        self.assertTrue(report["dense_prediction_check"]["passed"])
        self.assertEqual(sum(p.numel() for p in previous.parameters()),
                         sum(p.numel() for p in continued.parameters()))
        for key, value in continued.state_dict().items():
            self.assertTrue(torch.equal(value, previous.state_dict()[key]), key)


@unittest.skipUnless(WEIGHTS.with_name("yolo26s.pt").is_file(), "Local pretrained YOLO26s weights are required")
class SmallSportsBallTransferTests(SportsBallTransferTests):
    weights = WEIGHTS.with_name("yolo26s.pt")
    classifier_width = 128


if __name__ == "__main__":
    unittest.main()
