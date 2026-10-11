"""Train BPMP-Net on a server; run python train.py --help for configuration."""

import argparse
import csv
import json
from pathlib import Path
import random
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Directory containing pan/ms/label")
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Separate directory for each experiment")
    parser.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    parser.add_argument("--epochs", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Images per optimizer update")
    parser.add_argument("--epoch-mode", choices=("legacy", "full"), default="legacy")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--structure", choices=("ms-ssim", "ssim"), default="ms-ssim")
    parser.add_argument("--data-range", type=float, default=1.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Resume a portable state-dict checkpoint")
    group.add_argument("--pretrained", help="Load weights only; start a new optimizer")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "micro_batch_size", "val_every"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.workers < 0 or args.lr <= 0 or args.data_range <= 0:
        parser.error("workers must be nonnegative; lr and data-range must be positive")
    return args


def main():
    args = parse_args()
    # Imports follow argument parsing so --help also works before dependencies are installed.
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from bpmp.checkpoints import load_checkpoint, save_checkpoint
    from bpmp.data import PansharpeningDataset
    from bpmp.engine import train_epoch, validate
    from bpmp.losses import CompositeLoss
    from bpmp.models import BPMPNet

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this environment; use your server or --device cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PansharpeningDataset(args.train_dir), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        PansharpeningDataset(args.val_dir), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    model = BPMPNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = CompositeLoss(args.structure, args.data_range).to(device)
    start_epoch, best_loss = 1, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, device, optimizer)
        # Changing the objective or sampling semantics during resume is easy to miss.
        old_config = checkpoint.get("config", {})
        for key in ("structure", "data_range", "epoch_mode", "micro_batch_size", "batch_size"):
            if key in old_config and old_config[key] != getattr(args, key):
                raise ValueError(f"Resume setting {key} differs; use --pretrained for a new experiment")
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "loader_rng_state" in checkpoint:
            generator.set_state(checkpoint["loader_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
    elif args.pretrained:
        load_checkpoint(args.pretrained, model, device)
    if start_epoch > args.epochs:
        raise ValueError("Checkpoint already reached --epochs; increase the target epoch count")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = output / "history.csv"
    if not args.resume and (history.exists() or (output / "latest.pth").exists()):
        raise FileExistsError("Experiment output already exists; choose a new directory or --resume")
    config = vars(args)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epoch mode: {args.epoch_mode}; images/update: {args.micro_batch_size}; device: {device}")
    with history.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "val_loss", "samples", "updates", "seconds"))
        if history.stat().st_size == 0:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.perf_counter()
            stats = train_epoch(model, train_loader, optimizer, criterion, device, args.epoch_mode, args.micro_batch_size)
            val_loss = None
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val_loss = validate(model, val_loader, criterion, device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(output / "best.pth", model, optimizer, epoch, best_loss, config, generator)
                save_checkpoint(output / "latest.pth", model, optimizer, epoch, best_loss, config, generator)
            row = {"epoch": epoch, "train_loss": stats["loss"], "val_loss": val_loss,
                   "samples": stats["samples"], "updates": stats["updates"],
                   "seconds": time.perf_counter() - started}
            writer.writerow(row)
            handle.flush()
            print(f"Epoch {epoch}/{args.epochs}: train={stats['loss']:.8f}, val={val_loss}, updates={stats['updates']}")


if __name__ == "__main__":
    main()
