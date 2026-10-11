"""Server-side regression checks. No real dataset or training experiment needed."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch
from torch import nn
from pytorch_msssim import ms_ssim

from bpmp.checkpoints import load_checkpoint, save_checkpoint
from bpmp.data import PansharpeningDataset
from bpmp.engine import train_epoch, validate
from bpmp.losses import CompositeLoss
from bpmp.metrics import quality_index


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pan, ms):
        prediction = self.bn(pan) * self.scale
        return prediction, prediction


class RuntimeTests(unittest.TestCase):
    def test_composite_loss_matches_historical_dual_output_formula(self):
        torch.manual_seed(7)
        target = torch.rand(1, 3, 256, 256) * 2 - 1
        outputs = (target + 0.02, target - 0.03)
        expected = sum(
            0.6 * nn.functional.l1_loss(output, target)
            + 0.3 * nn.functional.mse_loss(output, target)
            + 0.1 * (1 - ms_ssim(output, target, data_range=1.0))
            for output in outputs
        )
        torch.testing.assert_close(CompositeLoss()(outputs, target), expected)

    def test_pairing_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder, shape in (("pan", (256, 256)), ("ms", (64, 64, 3)), ("label", (256, 256, 3))):
                (root / folder).mkdir()
                for name in ("b", "a"):
                    image = np.full(shape, 128, dtype=np.uint8)
                    self.assertTrue(cv2.imwrite(str(root / folder / f"{name}.png"), image))
            dataset = PansharpeningDataset(root)
            self.assertEqual(dataset.names, ["a", "b"])
            self.assertEqual(tuple(dataset[0]["ms"].shape), (3, 64, 64))
            self.assertAlmostEqual(dataset[0]["ms"][0, 0, 0].item(), (128 / 255 - 0.5) / 0.5, places=6)
            (root / "ms/a.png").rename(root / "ms/c.png")
            with self.assertRaises(ValueError):
                PansharpeningDataset(root)

    def test_short_batches_full_epoch_and_validation_mode_restore(self):
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = lambda outputs, target: sum(nn.functional.mse_loss(output, target) for output in outputs)
        batches = [
            {"pan": torch.randn(n, 1, 8, 8), "ms": torch.zeros(n, 1, 8, 8), "target": torch.zeros(n, 1, 8, 8)}
            for n in (2, 1)
        ]
        stats = train_epoch(model, batches, optimizer, criterion, "cpu", "full", 1)
        self.assertEqual(stats["samples"], 3)
        self.assertEqual(stats["updates"], 3)
        before = model.bn.num_batches_tracked.clone()
        validate(model, batches, criterion, "cpu")
        self.assertTrue(model.training)
        torch.testing.assert_close(model.bn.num_batches_tracked, before)
        stats = train_epoch(model, batches, optimizer, criterion, "cpu", "legacy", 1)
        self.assertEqual(stats["samples"], 2)
        self.assertEqual(stats["updates"], 2)
        self.assertEqual(model.bn.num_batches_tracked.item(), before.item() + 2)

    def test_portable_checkpoint_roundtrip(self):
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        generator = torch.Generator().manual_seed(9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            save_checkpoint(path, model, optimizer, 3, 0.1, {"seed": 9}, generator)
            restored = TinyModel()
            checkpoint = load_checkpoint(path, restored, optimizer=torch.optim.Adam(restored.parameters()))
            self.assertEqual(checkpoint["epoch"], 3)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)

    def test_quality_index_constant_limits(self):
        zeros, ones = torch.zeros(8, 8), torch.ones(8, 8)
        self.assertEqual(quality_index(zeros, zeros).item(), 1)
        self.assertEqual(quality_index(ones, ones).item(), 1)
        self.assertEqual(quality_index(zeros, ones).item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
