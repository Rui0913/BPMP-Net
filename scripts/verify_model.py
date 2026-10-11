"""Compare original and cleaned model outputs/gradients on your server."""

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backward", action="store_true", help="Also compare gradients and BatchNorm buffers")
    args = parser.parse_args()
    import torch
    from bpmp.models import BPMPNet
    sys.path.insert(0, str(ROOT / "legacy/prj3113"))
    original_class = importlib.import_module("Net2").Net
    device = torch.device(args.device)
    torch.manual_seed(123)
    original = original_class().to(device)
    torch.manual_seed(123)
    cleaned = BPMPNet().to(device)
    original_state, cleaned_state = original.state_dict(), cleaned.state_dict()
    if original_state.keys() != cleaned_state.keys():
        raise AssertionError("State-dict keys differ")
    for key in original_state:
        torch.testing.assert_close(original_state[key], cleaned_state[key], rtol=0, atol=0)
    original.eval()
    cleaned.eval()
    pan = torch.rand(1, 1, 256, 256, device=device) * 2 - 1
    ms = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    with torch.no_grad():
        before, after = original(pan, ms), cleaned(pan, ms)
        for reference, result in zip(before, after):
            torch.testing.assert_close(reference, result, rtol=1e-5, atol=1e-6)
            if result.shape != (1, 3, 256, 256):
                raise AssertionError(f"Unexpected output shape: {result.shape}")
    print("PASS: initialization, state-dict keys and both evaluation outputs agree")
    if args.backward:
        original.train()
        cleaned.train()
        sum(output.square().mean() for output in original(pan, ms)).backward()
        sum(output.square().mean() for output in cleaned(pan, ms)).backward()
        for (name, reference), (other_name, result) in zip(original.named_parameters(), cleaned.named_parameters()):
            if name != other_name or (reference.grad is None) != (result.grad is None):
                raise AssertionError(f"Gradient layout differs: {name}, {other_name}")
            if reference.grad is not None:
                torch.testing.assert_close(reference.grad, result.grad, rtol=1e-4, atol=1e-6)
        for key, reference in original.state_dict().items():
            torch.testing.assert_close(reference, cleaned.state_dict()[key], rtol=1e-5, atol=1e-6)
        print("PASS: training-mode gradients and BatchNorm buffer updates agree")


if __name__ == "__main__":
    main()
