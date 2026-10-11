"""Training and validation loops with explicit epoch and update semantics."""

import torch


def train_epoch(model, loader, optimizer, criterion, device, epoch_mode="legacy", micro_batch_size=1):
    """Train on one shuffled loader batch (legacy) or all batches (full).

    Each micro-batch causes an optimizer update; micro_batch_size=1 retains
    the historical per-image update/BatchNorm behavior. This is not gradient
    accumulation into a batch of 16. Restoring train() fixes the old loop's
    failure to resume BatchNorm updates after validation.
    """
    if epoch_mode not in {"legacy", "full"}:
        raise ValueError("epoch_mode must be legacy or full")
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")
    model.train()
    total_loss = 0.0
    samples = 0
    updates = 0
    for batch in loader:
        batch_size = batch["pan"].shape[0]
        for start in range(0, batch_size, micro_batch_size):
            stop = min(start + micro_batch_size, batch_size)
            pan = batch["pan"][start:stop].to(device, non_blocking=True)
            ms = batch["ms"][start:stop].to(device, non_blocking=True)
            target = batch["target"][start:stop].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(pan, ms), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; check input data and structural loss settings")
            loss.backward()
            optimizer.step()
            count = stop - start
            total_loss += loss.item() * count
            samples += count
            updates += 1
        if epoch_mode == "legacy":
            break
    if samples == 0:
        raise ValueError("Training loader is empty")
    return {"loss": total_loss / samples, "samples": samples, "updates": updates}


@torch.no_grad()
def validate(model, loader, criterion, device):
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    samples = 0
    try:
        for batch in loader:
            pan = batch["pan"].to(device, non_blocking=True)
            ms = batch["ms"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            loss = criterion(model(pan, ms), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            count = pan.shape[0]
            total_loss += loss.item() * count
            samples += count
    finally:
        model.train(previous_mode)
    if samples == 0:
        raise ValueError("Validation loader is empty")
    return total_loss / samples
