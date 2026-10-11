"""Export final HRMS predictions and optional reduced-reference metrics."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing pan/ms and optional label")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true", help="Requires label/ and writes metrics.csv")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")

    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from bpmp.checkpoints import load_checkpoint
    from bpmp.data import PansharpeningDataset
    from bpmp.models import BPMPNet

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use your server or --device cpu")
    model = BPMPNet().to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    loader = DataLoader(PansharpeningDataset(args.data_dir, require_target=args.evaluate),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            pan, ms = batch["pan"].to(device), batch["ms"].to(device)
            _, final = model(pan, ms)
            if not torch.isfinite(final).all():
                raise FloatingPointError("Non-finite prediction")
            # Export convention follows legacy clipping and truncation to uint8.
            image01 = final.clamp(-1, 1).add(1).mul(0.5).cpu()
            images = image01.mul(255).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            for index, name in enumerate(batch["name"]):
                image_path = output / f"{name}.png"
                if not cv2.imwrite(str(image_path), images[index]):
                    raise OSError(f"Could not write {image_path}")
                if args.evaluate:
                    from bpmp.metrics import evaluate_image
                    # Evaluate exported 8-bit pixels, rather than unquantized network outputs.
                    prediction = torch.from_numpy(images[index]).permute(2, 0, 1).unsqueeze(0).float().div(255)
                    target = batch["target"][index:index + 1].add(1).mul(0.5)
                    low_ms = batch["ms"][index:index + 1].add(1).mul(0.5)
                    input_pan = batch["pan"][index:index + 1].add(1).mul(0.5)
                    rows.append({"name": name, **evaluate_image(prediction, target, low_ms, input_pan)})
    if rows:
        keys = list(rows[0])
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"name": "MEAN", **{key: sum(row[key] for row in rows) / len(rows) for key in keys[1:]}})
    print(f"Exported {len(loader.dataset)} final predictions to {output}")


if __name__ == "__main__":
    main()
