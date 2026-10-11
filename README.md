# BPMP-Net

**Bidirectional Progressive Multiscale Pyramid Fusion Network for Pansharpening**

Code for **A progressive multiscale network for pansharpening with bidirectional pyramids**.

**Authors:** Rui Hao, Xin Jin, Shengfa Miao, Puming Wang, Hongyue Huang, Qian Jiang.  
**Journal:** Engineering Applications of Artificial Intelligence, Volume 182 (2026), Article 116059.  
**DOI:** `10.1016/j.engappai.2026.116059`  
**Paper:** [Read the paper via DOI](https://doi.org/10.1016/j.engappai.2026.116059)  
**Documentation:** [中文 README](README.zh-CN.md) · [Implementation notes](docs/REFACTOR_NOTES.md)

This release organizes the historical **prj3113/Net2.py + main3.py** experiment
into a readable model package, shared loss function, configurable training and
prediction entry points. The original files are preserved under `legacy/prj3113/`.
It preserves the historical model and parameter keys; it does not claim newly
verified reproduction of the published numerical results.

## Overview and contributions

Pansharpening combines a high-resolution panchromatic (PAN) image with a
low-resolution multispectral (MS) image to reconstruct a high-resolution
multispectral (HRMS) image. BPMP-Net aims to retain spectral information while
recovering spatial edges and textures. The paper evaluates the method on
Maryland, QuickBird and WorldView-2 datasets.

The paper introduces four complementary designs:

1. **Bidirectional progressive multiscale pyramid fusion.** PAN, MS and fusion
   branches exchange features across modalities and the internal 128 × 128 and
   256 × 256 scales. Bidirectional information flow combines coarse structural
   context with fine spatial details instead of relying on one-way pyramid fusion.
2. **MCAB — Multiscale Cross-Attention Block.** Cross-attention models PAN–MS
   correlations, while serial–parallel multiscale convolutions combine wider
   context and local detail to guide spatial–spectral fusion.
3. **SKAFE — Shuffle Selective Kernel Attention Feature Extraction.** Self-attention
   and selective kernel attention aggregate multiscale features. Feature shuffling
   promotes information exchange, and progressive recovery preserves source-image
   structure while refining coarse features and fine details.
4. **SCWAB — Spatial-Channel Window Self-Attention Block.** Spatial and channel
   attention are coupled with regular/shifted window attention and dense feature
   connections. Local windows model contextual dependencies at a lower
   computational cost than global self-attention, while channel recalibration
   refines the spectral representation.

| Paper component | Implementation |
|---|---|
| Bidirectional pyramid and three-branch fusion | [`BPMPNet`](bpmp/models/network.py) |
| MCAB | [`MCAB`](bpmp/models/network.py) |
| SKAFE | [`SKAFE`](bpmp/models/skafe.py) |
| SCWAB | [`SCWAB`](bpmp/models/scwab.py) |
| Regular/shifted window attention | [`window_attention.py`](bpmp/models/window_attention.py) |
| Composite supervision | [`CompositeLoss`](bpmp/losses.py) |

These descriptions summarize the paper's design. The selected historical code's
output-scale and structural-loss conventions are documented below.

## Project layout

```text
bpmp/
  models/
    network.py          BPMPNet, PAN branch, MCAB
    skafe.py            SKAFE and shuffle selective attention
    scwab.py            spatial/channel/window attention cascade
    window_attention.py shared regular/shifted window attention
    common.py           convolution and residual blocks
  data.py               name-based PAN/MS/GT pairing and normalization
  losses.py             shared 0.6 L1 + 0.3 MSE + 0.1 structure loss
  engine.py             training and validation loops
  checkpoints.py        portable state-dict checkpoints
  metrics.py            per-image evaluation conventions
train.py                training CLI
predict.py              final HRMS export and optional metrics
scripts/                static checks, model parity, checkpoint conversion
tests/                  server-side regression checks
legacy/prj3113/         pristine reference files, not the default entry points
docs/                   provenance, changes, validation status
```

## Server environment

Clone the repository and enter its root:

```bash
git clone https://github.com/Rui0913/BPMP-Net.git
cd BPMP-Net
```

While the code/documentation PR is open, run `git checkout refactor/prj3113-readable`
to use this release. After it is merged, the same files are available on `main`.

Use Python 3.10 or newer. Install a PyTorch build appropriate for the server's
CUDA/driver environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Minimum dependency versions describe the APIs used by this release; they are
not the original experiment's environment lock. Save your actual package
versions with the experiment. Training, inference, throughput and numerical
reproduction must be validated on the server. No dataset or trained weights
are included in this repository.

## Data layout and preprocessing

```text
dataset/
  train/
    pan/sample001.png    256 x 256, grayscale, uint8
    ms/sample001.png      64 x 64, three BGR channels, uint8
    label/sample001.png  256 x 256, three BGR channels, uint8
  val/
    pan/  ms/  label/
  test/
    pan/  ms/  label/     label is optional when only exporting predictions
```

Images are paired by filename stem and checked for missing/duplicate pairs.
Different suffixes may share the same stem across modalities. Inputs use the
historical OpenCV **BGR** channel order and normalization `(pixel / 255 - 0.5) / 0.5`.
Images are not silently resized. The model supports this three-channel image
pipeline, not raw four-band QuickBird or eight-band WorldView-2 products.

## Training

```bash
python train.py \
  --train-dir /data/QB/train \
  --val-dir /data/QB/val \
  --output-dir runs/qb_legacy \
  --device cuda:0
```

Defaults from the selected experiment: Adam, learning rate `0.001`, loader batch
size `16`, `11000` iterations called epochs, validation every `50` epochs, two
output losses weighted `0.6/0.3/0.1`.

Two settings make the original update schedule explicit:

- `--epoch-mode legacy` (default): only the first shuffled loader batch is used
  in each epoch, matching the historical sampling schedule.
- `--micro-batch-size 1` (default): each image causes a separate optimizer update,
  matching the historical loop. This is not a single batch-of-16 update.

For conventional full-dataset epochs, choose `--epoch-mode full`. To update once
per 16 images, additionally choose `--micro-batch-size 16`. These changes affect
the number of updates and BatchNorm behavior; `11000` full epochs is not the
same training budget as `11000` historical iterations. Choose a new budget and
output directory for that experiment.

The cleaned loop fixes name pairing, short final batches, seed overriding, CPU
selection and the failure to restore training mode after validation. These fixes
can change a retrained model even with the historical sampling schedule. See
[the change record](docs/REFACTOR_NOTES.md).

## Loss and the paper/code differences

Section 3.6 of the paper defines a composite objective at each supervised scale:

```text
L_s = 0.6 * L1(y_s, prediction_s)
    + 0.3 * MSE(y_s, prediction_s)
    + 0.1 * (1 - SSIM(y_s, prediction_s))
L_total = sum(L_s for s in {128, 256})
```

L1 emphasizes pixel-level radiometric and spectral consistency, MSE penalizes
large reconstruction errors, and SSIM encourages structural agreement.

For each output, the selected historical code actually computes:

```text
L = 0.6 * L1 + 0.3 * MSE + 0.1 * (1 - MS-SSIM)
L_total = L(auxiliary_hrms, GT) + L(final_hrms, GT)
```

Both historical outputs are **256 x 256**, even though the features interact at
128 and 256 pixels. The auxiliary head is placed after upsampling. This differs
from the paper's description of separately supervising 128/256 output scales.

The original call is `pytorch_msssim.ms_ssim(..., data_range=1.0)` on tensors
normalized to `[-1, 1]`; the paper names SSIM. Defaults retain this exact function
and range. To investigate a single-scale SSIM objective with the actual input
range, explicitly use `--structure ssim --data-range 2.0`. This is a new loss
configuration, not a claim of historical-result equivalence. No output is resized
or new loss scale inserted to conceal the discrepancy.

## Checkpoints and inference

Training writes `config.json`, sample-weighted `history.csv`, `best.pth` and
`latest.pth`. Both checkpoint files store state dicts, optimizer state and epoch;
the best checkpoint is selected by validation loss. Latest is saved at validation
intervals and on the last epoch.

```bash
python train.py \
  --train-dir /data/QB/train --val-dir /data/QB/val \
  --output-dir runs/qb_legacy --device cuda:0 \
  --resume runs/qb_legacy/latest.pth

python predict.py \
  --data-dir /data/QB/test \
  --checkpoint runs/qb_legacy/best.pth \
  --output-dir results/qb --device cuda:0 --evaluate
```

Inference exports the **second/final** output, clips to `[-1, 1]`, maps to `[0, 255]`
and truncates to uint8 PNG. `--evaluate` requires ground truth and writes per-image
and mean metrics to `metrics.csv`; omit it when GT is unavailable. Metrics are
computed on exported 8-bit pixels. SSIM is single-scale, SAM uses radians, UIQI
uses valid 32-pixel windows, and QNR uses normalized distortion averages with
bicubic PAN degradation. This is not a sensor-specific full-resolution MTF
assessment. It differs from the legacy unnormalized sums/absolute-value QNR;
use the archived metric script when studying those historical conventions.

Old `model_state_dict` files retain their parameter keys. Whole-model pickle
checkpoints need an explicit conversion, performed only on trusted files:

```bash
python -m pip install -r requirements-legacy.txt
python scripts/convert_checkpoint.py \
  --input /path/to/model_best_epoch.pth \
  --output converted/best.pth --allow-pickle
```

For a state-dict checkpoint omit `--allow-pickle`. Conversion rejects single-output
and ablation checkpoints whose parameter keys differ. Normal training/prediction
uses `weights_only=True` and never imports a pickled model automatically.

## Verification

Dependency-free source checks:

```bash
python scripts/check_source.py
```

On the server, after installing dependencies:

```bash
python -m unittest discover -s tests -v
python -m pip install -r requirements-legacy.txt
python scripts/verify_model.py --device cuda:0 --backward
```

The static check verifies syntax, archived source hashes, renamed model computation
ASTs and CLI help. Runtime checks cover the loss, pairing, partial batches,
validation/train mode, checkpoints, original-vs-clean initialization, outputs and
gradients. See [validation status](docs/VALIDATION.md) for what has actually run.

## Citation

```bibtex
@article{hao2026bpmp,
  title = {A progressive multiscale network for pansharpening with bidirectional pyramids},
  author = {Hao, Rui and Jin, Xin and Miao, Shengfa and Wang, Puming and Huang, Hongyue and Jiang, Qian},
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {182},
  pages = {116059},
  year = {2026},
  doi = {10.1016/j.engappai.2026.116059}
}
```
