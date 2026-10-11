"""Paired 8-bit BGR images with the original [-1, 1] normalization."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_index(folder: Path) -> dict[str, Path]:
    """Index by stem so a.png and a.tif can be paired across modalities."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing image directory: {folder}")
    index = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in index:
                raise ValueError(f"Duplicate image stem {path.stem!r} in {folder}")
            index[path.stem] = path
    if not index:
        raise ValueError(f"No images found in {folder}")
    return index


def read_tensor(path: Path, channels: int, size: tuple[int, int]) -> torch.Tensor:
    """Read without implicit resizing, RGB conversion, or 16-bit truncation."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected 8-bit image, got {image.dtype}: {path}")
    if channels == 1:
        # Match the legacy cv2.imread(flags=0) grayscale conversion exactly.
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot decode grayscale image: {path}")
        image = image[:, :, None]
    if image.shape != (*size, channels):
        raise ValueError(f"Expected {(*size, channels)}, got {image.shape}: {path}")
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    # Operation order matches torchvision ToTensor followed by Normalize.
    tensor = torch.from_numpy(image).to(torch.float32).div(255.0)
    return tensor.sub(0.5).div(0.5)


class PansharpeningDataset(Dataset):
    """Pair PAN/MS/GT by image stem, rather than unrelated filesystem order.

    Layout: root/pan, root/ms, root/label. Inference can omit root/label.
    This version supports three BGR channels, not raw four/eight-band TIFFs.
    """

    def __init__(self, root: str | Path, require_target: bool = True):
        root = Path(root)
        self.pan = image_index(root / "pan")
        self.ms = image_index(root / "ms")
        self.targets = image_index(root / "label") if require_target else None
        expected = set(self.pan)
        for name, index in (("ms", self.ms), ("label", self.targets)):
            if index is not None and set(index) != expected:
                missing = sorted(expected - set(index))[:5]
                extra = sorted(set(index) - expected)[:5]
                raise ValueError(f"Unpaired {name} images: missing={missing}, extra={extra}")
        self.names = sorted(expected)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        sample = {
            "pan": read_tensor(self.pan[name], 1, (256, 256)),
            "ms": read_tensor(self.ms[name], 3, (64, 64)),
            "name": name,
        }
        if self.targets is not None:
            sample["target"] = read_tensor(self.targets[name], 3, (256, 256))
        return sample
