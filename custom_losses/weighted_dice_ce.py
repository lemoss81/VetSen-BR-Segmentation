"""Vendored copy of WeightedDiceCrossEntropyLoss with `class_weights` support.

The upstream pytorch_segmentation_models_trainer package (pinned in pixi.toml)
does not expose `class_weights` on WeightedDiceCrossEntropyLoss -- that's a
customization that only exists in the OLD_IP project's local/modified copy of
the library. Rather than depend on a private fork, this module reproduces
that one class locally so configs/loss/weighted_ce.yaml can target it
directly. Everything else (dice/CE mix, smoothing, ignore_index) matches the
upstream class exactly; only `class_weights` is added.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class WeightedDiceCrossEntropyLoss(nn.Module):
    """
    Combined Cross Entropy and Dice Loss for segmentation tasks.

    Args:
        num_classes (int): Number of classes in the segmentation task
        dice_weight (float): Weight for dice loss (default: 0.75)
        ce_weight (float): Weight for cross entropy loss (default: 0.25)
        smooth_factor (float): Smoothing factor for cross entropy loss (default: 0)
        ignore_index (int): Label value to ignore in loss computation (default: None)
        class_weights (list[float], optional): Per-class weights passed to
            CrossEntropyLoss. Length must match num_classes.
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.75,
        ce_weight: float = 0.25,
        smooth_factor: float = 0.0,
        ignore_index: int = None,
        class_weights=None,
    ):
        super(WeightedDiceCrossEntropyLoss, self).__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index if ignore_index is not None else 255

        ce_class_weights = None
        if class_weights is not None:
            ce_class_weights = torch.as_tensor(list(class_weights), dtype=torch.float32)
            if ce_class_weights.numel() != num_classes:
                raise ValueError(
                    "class_weights length must match num_classes "
                    f"({ce_class_weights.numel()} != {num_classes})"
                )

        mode = "multiclass" if num_classes > 2 else "binary"

        self.dice_loss = smp.losses.DiceLoss(
            mode=mode,
            ignore_index=self.ignore_index,
        )

        self.ce_loss = nn.CrossEntropyLoss(
            weight=ce_class_weights,
            label_smoothing=smooth_factor,
            ignore_index=self.ignore_index,
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        combined = self.dice_weight * dice + self.ce_weight * ce
        return combined
