# Refactor decisions and provenance

Source: `prj3113/Net2.py` and `main3.py`, with their direct model/data dependencies.
The neighboring `prj3116/Net.py` has the same model source, but its current
dual-output training entry uses 0.7/0.2/0.1 instead of 0.6/0.3/0.1.
Nine selected historical files are archived byte-for-byte. The old experimental
variants and hard-coded prediction scripts are not the new project's entry points.

## Model changes limited to readability

- Rename public classes to BPMPNet, MCAB, SKAFE and SCWAB.
- Split modules by responsibility, use explicit relative imports, rename local
  intermediate tensors, remove commented experiments and unused imports.
- Reuse the identical window-attention implementation formerly duplicated in
  CAB.py and MSA.py. AST equality was checked before deduplication.
- Preserve registered submodule names, tensor operations, shared weights,
  initialization order and BatchNorm updates. Whole-model pickle module paths
  change; a separate converter resolves the archived module names.

The following quirks are deliberately retained, not silently repaired:

- The cross-modal module evaluates three pairs on the same inputs and returns
  only the final pair. It does not feed earlier outputs to later pairs.
- The selective-attention convolution sum is used for shape extraction while
  the split gates operate on the original input. Computing it still updates
  BatchNorm buffers in training mode.
- SKAFE computes a 13-kernel branch but excludes that branch from the output sum.
- Registered unused layers, repeated residual normalization/activation objects,
  and shared MCAB/SCWAB calls remain. Removing them could change initialization,
  checkpoint keys, gradients or BatchNorm state.
- The PAN residual operates on projected features and contains one explicit
  residual block followed by convolution/extraction; the paper's compact two-RB
  equation is not implemented literally.
- Both supervised outputs are 256 pixels. The historical structural term is
  MS-SSIM with data_range=1 on [-1,1] tensors.

These are reasons to retain a known source model and run controlled server
experiments before undertaking architectural/performance changes.

## Training behavior changes

1. Pair modalities by matching stems, reject missing/duplicate pairs, validate
   dtype and image shape instead of reshaping arbitrary arrays.
2. Restore training mode after validation and at the start of every epoch.
   The original model stayed in evaluation mode after the first validation.
3. Use one composite loss implementation in both loops. Preserve historical
   weights, function and data range by default; expose alternative SSIM explicitly.
4. Handle partial batches, make per-image optimizer updates explicit, and offer
   full-dataset epochs separately. Default historical sampling consumes one
   shuffled loader batch per iteration; validation processes the whole set.
5. Honor a specified seed instead of overwriting it with a random number.
   Validation no longer shuffles; loss reporting is weighted by image count.
6. Select CPU/CUDA explicitly; remove globals, redundant initial loader scans,
   nonfunctional CUDA prefetching and checkpoint reloads within training.
7. Save state dicts atomically with optimizer, configuration, best loss and RNG
   states. Do not automatically unpickle whole model objects.
8. Export the actual final output for each sample without squeezing away batch
   axes, hard-coded model paths, filenames represented as lists or fixed test count.

The cleaned trainer is a corrected continuation of the source experiment, not a
bit-for-bit replay of its data-order/training-mode bugs. A freshly trained model
must be re-evaluated on the server. Enabling full epochs, larger micro-batches,
single-scale SSIM or a different data range constitutes another experiment.

## Metrics

The legacy evaluator is retained unchanged for provenance. The cleaned evaluator
uses per-image metrics on quantized final BGR output in [0,1]; SSIM is from
pytorch-msssim, SAM is in radians, UIQI averages valid local windows. Global Q
defines constant-image limits. D_lambda and D_s are averaged over band pairs and
bands, respectively; QNR is (1-D_lambda)(1-D_s), without the legacy absolute value.
PAN degradation remains bicubic. No sensor-specific MTF preprocessing is implied.

Do not compare these new metrics to the paper as proof of reproduction without
aligning preprocessing, checkpoint selection, test splits and metric conventions.
