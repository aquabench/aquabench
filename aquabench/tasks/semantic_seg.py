"""Semantic-segmentation task: CrossEntropy loss + mean-IoU metric.

The task owns the loss (so every head trains with the same CE objective — heads
return logits only) and the metric. A future classification task would slot in by
defining its own `loss` / `build_metric` here.
"""

import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex


class _ExcludeClassMacroJaccard(Metric):
    """Macro mIoU that excludes `exclude_class` from the class-level average,
    but keeps its pixels in the confusion matrix.

    MulticlassJaccardIndex(ignore_index=k) drops all pixels where target==k,
    so false positives on those pixels vanish and IoU degenerates to recall for
    binary datasets.  This wrapper computes per-class IoU with all pixels, then
    excludes class k only from the macro average.
    """

    def __init__(self, num_classes: int, exclude_class: int):
        super().__init__()
        self._inner = MulticlassJaccardIndex(num_classes=num_classes, average=None)
        self.exclude_class = exclude_class

    def update(self, preds, target):
        self._inner.update(preds, target)

    def compute(self):
        per_class = self._inner.compute()  # [num_classes]
        keep = [i for i in range(per_class.shape[0]) if i != self.exclude_class]
        return per_class[keep].nanmean()

    def reset(self):
        self._inner.reset()


class SemanticSegTask:
    name = "semantic_seg"
    metric_name = "miou"

    def __init__(self, num_classes: int, ignore_index: int, loss_ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index            # metric ignore (background)
        self.loss_ignore_index = loss_ignore_index  # CE ignore (unlabeled)

    # ---- loss (CE for all heads) ----
    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels, ignore_index=self.loss_ignore_index)

    # ---- metric (IoU; logged at val/test only) ----
    def build_metric(self, include_background: bool = False):
        # include_background=True  -> macro over all classes (including background).
        # include_background=False -> macro over non-background classes, but background
        #   pixels still enter the confusion matrix so FP on background are counted.
        #   (Using ignore_index in MulticlassJaccardIndex drops background pixels
        #   entirely, making FP=0 and collapsing IoU to recall for binary datasets.)
        if include_background:
            return MulticlassJaccardIndex(num_classes=self.num_classes, average="macro")
        return _ExcludeClassMacroJaccard(
            num_classes=self.num_classes, exclude_class=self.ignore_index
        )

    def per_class_metric(self):
        # average=None returns per-class IoU; all pixels in confusion matrix so FP count.
        return MulticlassJaccardIndex(
            num_classes=self.num_classes, average=None
        )

    def step(self, model, batch):
        """Return (loss, preds[B,H,W], target[B,H,W]).

        Heads return logits at their native (token/decoder) resolution; the task
        upsamples them to the label resolution before loss/metric (DINOv3-style
        for the linear probe: predict at the token grid, upsample the logits).
        """
        labels = batch["labels"]
        logits = model(batch["pixel_values"])
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        return self.loss(logits, labels), logits.argmax(1), labels
