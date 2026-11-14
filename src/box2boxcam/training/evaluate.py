"""Evaluate COCO sports-ball and fine-tuned detectors against identical labels."""

from collections import Counter

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix


class BallValidator(DetectionValidator):
    """Map sports ball predictions to ball=0, never filter ground-truth labels."""

    def init_metrics(self, model):
        names = model.names
        candidates = [int(key) for key, name in names.items() if name in {"sports ball", "ball"}]
        if len(candidates) != 1:
            raise ValueError(f"Expected one ball class, got {names}")
        self.ball_class = candidates[0]
        super().init_metrics(model)
        self.names = {0: "ball"}
        self.nc = 1
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=self.names)
        self.fixed = Counter()

    def postprocess(self, preds):
        results = super().postprocess(preds)
        for index, pred in enumerate(results):
            selected = pred["cls"] == self.ball_class
            results[index] = {key: value[selected] for key, value in pred.items()}
            results[index]["cls"].zero_()
        return results

    def update_metrics(self, preds, batch):
        # Fixed operating point: same confidence and IoU for both models.
        for index, pred in enumerate(preds):
            target = self._prepare_batch(index, batch)
            selected = pred["conf"] >= 0.25
            filtered = {key: value[selected] for key, value in pred.items()}
            tp = int(self._process_batch(filtered, target)["tp"][:, 0].sum())
            predicted = len(filtered["cls"])
            actual = len(target["cls"])
            self.fixed.update(tp=tp, fp=predicted - tp, fn=actual - tp, images=1)
            if actual == 0:
                self.fixed.update(negative_images=1, fp_on_negatives=predicted)
        super().update_metrics(preds, batch)

    def finalize_metrics(self):
        super().finalize_metrics()
        counts = dict(self.fixed)
        tp, fp, fn = (self.fixed[key] for key in ("tp", "fp", "fn"))
        counts.update(
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else 0.0,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            confidence=0.25, matching_iou=0.5,
        )
        self.metrics.fixed_threshold = counts
