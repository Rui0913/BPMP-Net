"""Per-image metrics on BGR tensors in [0, 1], with SAM in radians.

QNR uses normalized D_lambda/D_s and bicubic PAN degradation. This is an
explicit evaluation convention, not a sensor-specific MTF/Wald pipeline.
Historical unnormalized distortion sums remain in legacy/prj3113.
"""

import torch
from torch.nn import functional as F
from pytorch_msssim import ssim


def quality_index(a, b):
    """Global universal quality index with defined constant-image limits."""
    a, b = a.flatten(), b.flatten()
    mean_a, mean_b = a.mean(), b.mean()
    da, db = a - mean_a, b - mean_b
    var_a, var_b = da.square().mean(), db.square().mean()
    mean_sum = mean_a.square() + mean_b.square()
    var_sum = var_a + var_b
    eps = torch.finfo(a.dtype).eps
    luminance = torch.where(mean_sum > eps, 2 * mean_a * mean_b / mean_sum.clamp_min(eps), torch.ones_like(mean_sum))
    contrast = torch.where(var_sum > eps, 2 * (da * db).mean() / var_sum.clamp_min(eps), torch.ones_like(var_sum))
    return luminance * contrast


def uiqi(prediction, target, window_size=32):
    """Mean local Q over all valid sliding windows and channels."""
    mean_p = F.avg_pool2d(prediction, window_size, stride=1)
    mean_t = F.avg_pool2d(target, window_size, stride=1)
    var_p = (F.avg_pool2d(prediction.square(), window_size, stride=1) - mean_p.square()).clamp_min(0)
    var_t = (F.avg_pool2d(target.square(), window_size, stride=1) - mean_t.square()).clamp_min(0)
    covariance = F.avg_pool2d(prediction * target, window_size, stride=1) - mean_p * mean_t
    mean_sum, var_sum = mean_p.square() + mean_t.square(), var_p + var_t
    eps = torch.finfo(prediction.dtype).eps
    luminance = torch.where(mean_sum > eps, 2 * mean_p * mean_t / mean_sum.clamp_min(eps), torch.ones_like(mean_sum))
    contrast = torch.where(var_sum > eps, 2 * covariance / var_sum.clamp_min(eps), torch.ones_like(var_sum))
    return (luminance * contrast).mean()


@torch.no_grad()
def evaluate_image(prediction, target, ms, pan):
    """Evaluate one image, retaining CHW axes even when batch size is one."""
    if prediction.shape != (1, 3, 256, 256) or target.shape != prediction.shape:
        raise ValueError("Expected one prediction/target with shape (1,3,256,256)")
    if ms.shape != (1, 3, 64, 64) or pan.shape != (1, 1, 256, 256):
        raise ValueError("Expected MS (1,3,64,64) and PAN (1,1,256,256)")
    eps = torch.finfo(prediction.dtype).eps
    mse = F.mse_loss(prediction, target)
    psnr = -10 * torch.log10(mse)
    p, t = prediction.flatten(2), target.flatten(2)
    dp, dt = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
    cc_denominator = dp.square().sum(-1).sqrt() * dt.square().sum(-1).sqrt()
    cc = ((dp * dt).sum(-1) / cc_denominator.clamp_min(eps)).clamp(-1, 1).mean()
    norm_p, norm_t = prediction.norm(dim=1), target.norm(dim=1)
    norm_product = norm_p * norm_t
    cosine = ((prediction * target).sum(1) / norm_product.clamp_min(eps)).clamp(-1, 1)
    angles = cosine.acos()
    angles = torch.where((norm_p == 0) & (norm_t == 0), torch.zeros_like(angles), angles)
    sam = angles.mean()
    band_mse = (p - t).square().mean(-1)
    ergas = 25 * (band_mse / t.mean(-1).square().clamp_min(eps)).mean().sqrt()
    spectral_differences = []
    for i in range(3):
        for j in range(i + 1, 3):
            spectral_differences.append((quality_index(prediction[:, i], prediction[:, j]) - quality_index(ms[:, i], ms[:, j])).abs())
    d_lambda = torch.stack(spectral_differences).mean()
    low_pan = F.interpolate(pan, scale_factor=0.25, mode="bicubic", align_corners=False, recompute_scale_factor=True)
    d_s = torch.stack([
        (quality_index(prediction[:, i], pan[:, 0]) - quality_index(ms[:, i], low_pan[:, 0])).abs()
        for i in range(3)
    ]).mean()
    values = {
        "PSNR": psnr, "SSIM": ssim(prediction, target, data_range=1.0),
        "CC": cc, "SAM_rad": sam, "ERGAS": ergas, "UIQI": uiqi(prediction, target),
        "D_lambda": d_lambda, "D_s": d_s, "QNR": (1 - d_lambda) * (1 - d_s),
    }
    return {key: float(value.item()) for key, value in values.items()}
