"""Shared training/validation objective from prj3113/main3.py."""

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_msssim import ms_ssim, ssim


class CompositeLoss(nn.Module):
    """Sum 0.6 L1 + 0.3 MSE + 0.1 (1 - structural similarity) over outputs.

    Defaults preserve historical MS-SSIM(data_range=1.0) on [-1, 1] tensors.
    Paper-style single-scale SSIM and data_range=2.0 are explicit options;
    switching them changes the objective and requires a new server experiment.
    No output or target is silently resized to match the paper description.
    """

    def __init__(self, structure: str = "ms-ssim", data_range: float = 1.0):
        super().__init__()
        if structure not in {"ms-ssim", "ssim"}:
            raise ValueError(f"Unknown structure loss: {structure}")
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        self.structure = structure
        self.data_range = data_range

    def forward(self, outputs, target):
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if not outputs:
            raise ValueError("At least one prediction is required")
        structural_similarity = ms_ssim if self.structure == "ms-ssim" else ssim
        losses = []
        for output in outputs:
            if output.shape != target.shape:
                raise ValueError(f"Prediction {output.shape} and target {target.shape} differ")
            structure = structural_similarity(output, target, data_range=self.data_range)
            losses.append(
                0.6 * F.l1_loss(output, target)
                + 0.3 * F.mse_loss(output, target)
                + 0.1 * (1 - structure)
            )
        return torch.stack(losses).sum()
