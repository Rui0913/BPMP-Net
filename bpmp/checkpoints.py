"""Portable state-dict checkpoints; no automatic loading of pickled models."""

from pathlib import Path

import torch


def load_checkpoint(path, model, device="cpu", optimizer=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Expected a state-dict checkpoint; use scripts/convert_checkpoint.py for legacy models")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Checkpoint has no optimizer state; use --pretrained instead of --resume")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_checkpoint(path, model, optimizer, epoch, best_loss, config, generator):
    """Store weights, optimizer and RNG states for repeatable epoch-boundary resumes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "loader_rng_state": generator.get_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
