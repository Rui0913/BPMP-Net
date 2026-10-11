"""Convert historical whole-model checkpoints into portable state dicts."""

import argparse
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-pickle", action="store_true",
                        help="Only for your own trusted whole-model checkpoint; pickle executes code")
    args = parser.parse_args()
    import torch
    from bpmp.models import BPMPNet

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    if args.allow_pickle:
        # Old checkpoints contain references to Net/Net2/CAB/MSA/SCExt/base_net.
        legacy = Path(__file__).resolve().parents[1] / "legacy" / "prj3113"
        sys.path.insert(0, str(legacy))
        module = importlib.import_module("Net2")
        sys.modules["Net"] = module
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=not args.allow_pickle)
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model" in checkpoint and args.allow_pickle:
        state = checkpoint["model"].state_dict()
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state = checkpoint
    else:
        raise ValueError("Unsupported checkpoint; expected model_state_dict or trusted model object")
    # Strict load rejects weights from single-output/ablation models.
    BPMPNet().load_state_dict(state, strict=True)
    converted = {"format_version": 1, "model_state_dict": state, "epoch": checkpoint.get("epoch", 0)}
    for key in ("optimizer_state_dict", "best_loss", "config"):
        if key in checkpoint:
            converted[key] = checkpoint[key]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    print(f"Converted checkpoint: {output}")


if __name__ == "__main__":
    main()
