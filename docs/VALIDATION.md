# Validation status

## Local checks (2026-10-11)

Local checking uses the bundled Python interpreter and does not install or alter
the research training environment. This machine does not have PyTorch, OpenCV,
einops or pytorch-msssim in that runtime.

The dependency-free check is `python scripts/check_source.py`. It validates:

- All Python files parse and compile, including archived reference scripts.
- The nine archived source SHA256 hashes match the provenance manifest.
- Model definitions match the original AST after reversing documented class/local
  renames and removing docstrings and one unused SciPy import.
- Training, prediction and checkpoint-conversion CLIs support `--help` without
  importing research dependencies.

Result: **PASS** — 27 Python files compile; nine archived sources match SHA256;
28 model definitions retain their computation AST; CLI help works.
This is static evidence about model computation, not numerical output equivalence.

## Checks prepared for the server, not executed locally

```bash
python -m unittest discover -s tests -v
python scripts/verify_model.py --device cuda:0 --backward
```

The model comparison checks initialization/state-dict compatibility, both outputs,
training gradients and BatchNorm buffers against the archived model. It requires
the optional legacy dependencies. Regression tests cover loss equivalence, image
pairing/normalization, partial batches, epoch sampling, validation mode restoration,
state-dict serialization and constant-image Q limits.

No real-data training, server inference, GPU memory/throughput benchmark or
reproduction of the published PSNR/QNR tables has run in this task. Record those
results with dataset/checkpoint identifiers and the server environment before
claiming numerical reproduction.
