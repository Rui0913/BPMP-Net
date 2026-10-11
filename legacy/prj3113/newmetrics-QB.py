'''
评价指标从空间和光谱的角度来看，又可以分为，
1）光谱指标，包括SAM、ERGAS、RASE、(CC)、SSIM、Q、Q4、SID、QAVE；
2）空间指标，包括SCC、E、gradient、FCC、H、Q_F、RMSE、MSE、(CC)
CC在2篇论文中有歧义。一则说是光谱指标，一则说是空间指标。
'''
import math
import os
import cv2
import numpy as np
import xlwt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from typing import Tuple, List
import kornia


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [(k - 1) // 2 for k in kernel_size]
    return [computed[1], computed[1], computed[0], computed[0]]

def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input.device).to(input.dtype)
    tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # convolve the tensor with the kernel
    return F.conv2d(input_pad, tmp_kernel, padding=0, stride=1, groups=c)

def _compute_zero_padding(kernel_size: int) -> int:
    """Computes zero padding."""
    return (kernel_size - 1) // 2

def gaussian(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    return gauss / gauss.sum()

def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d

def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d

class SSIM(nn.Module):
    r"""Creates a criterion that measures the Structural Similarity (SSIM)
    index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    the loss, or the Structural dissimilarity (DSSIM) can be finally described
    as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Arguments:
        window_size (int): the size of the kernel.
        max_val (float): the dynamic range of the images. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.

    Returns:
        Tensor: the ssim index.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

    Examples::

        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = kornia.losses.SSIM(5, reduction='none')
        >>> loss = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(
            self,
            window_size: int,
            reduction: str = "none",
            max_val: float = 1.0) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.window = self.window.requires_grad_(False)  # need to disable gradients

        self.padding: int = _compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    def forward(  # type: ignore
            self,
            img1: torch.Tensor,
            img2: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(img1):
            raise TypeError("Input img1 type is not a torch.Tensor. Got {}"
                            .format(type(img1)))

        if not torch.is_tensor(img2):
            raise TypeError("Input img2 type is not a torch.Tensor. Got {}"
                            .format(type(img2)))

        if not len(img1.shape) == 4:
            raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}"
                             .format(img1.shape))

        if not len(img2.shape) == 4:
            raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}"
                             .format(img2.shape))

        if not img1.shape == img2.shape:
            raise ValueError("img1 and img2 shapes must be the same. Got: {}"
                             .format(img1.shape, img2.shape))

        if not img1.device == img2.device:
            raise ValueError("img1 and img2 must be in the same device. Got: {}"
                             .format(img1.device, img2.device))

        if not img1.dtype == img2.dtype:
            raise ValueError("img1 and img2 must be in the same dtype. Got: {}"
                             .format(img1.dtype, img2.dtype))

        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)

        # compute local mean per channel
        mu1: torch.Tensor = filter2D(img1, tmp_kernel)
        mu2: torch.Tensor = filter2D(img2, tmp_kernel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
        sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
        sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2

        ssim_map = ((2. * mu1_mu2 + self.C1) * (2. * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(ssim_map, min=0, max=1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            pass
        return loss


def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Function that computes PSNR

    See :class:`~kornia.losses.PSNR` for details.
    """
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")
    mse_val = mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input.device).to(input.dtype)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)


def ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int,
        reduction: str = "mean",
        max_val: float = 1.0) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.

    See :class:`~kornia.losses.SSIM` for details.
    """
    return SSIM(window_size, reduction, max_val)(img1, img2)

def ergas(img_fake, img_real, scale=4):
    """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""

    N, C, H, W = img_real.shape
    means_real = img_real.reshape(N, C, -1).mean(dim=-1)
    mses = ((img_fake - img_real) ** 2).reshape(N, C, -1).mean(dim=-1)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS

    return 100 / scale * torch.sqrt((mses / (means_real ** 2 + eps)).mean())

def sam(img1, img2):
    """SAM for (N, C, H, W) image; torch.float32 [0.,1.]."""
    inner_product = (img1 * img2).sum(dim=1)
    img1_spectral_norm = torch.sqrt((img1**2).sum(dim=1))
    img2_spectral_norm = torch.sqrt((img2**2).sum(dim=1))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + eps)).clamp(min=0, max=1)
    cos_theta = cos_theta.reshape(cos_theta.shape[0], -1)
    return torch.mean(torch.acos(cos_theta), dim=-1).mean()

eps = torch.finfo(torch.float32).eps
def cc(img1, img2):
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
                eps + torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean(dim=-1)


def uiqi(img1, img2, block_size=32):  # universal image quality index

    window = torch.ones((block_size, block_size)).unsqueeze(0) / (block_size ** 2)

    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = kornia.filters.filter2d(img1, window)[:, :, pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]

    mu2 = kornia.filters.filter2d(img2, window)[:, :, pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = kornia.filters.filter2d(img1 ** 2, window)[:, :, pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = kornia.filters.filter2d(img2 ** 2, window)[:, :, pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu2_sq
    #    print(mu1_mu2.shape)
    # print(sigma2_sq.shape)
    sigma12 = kornia.filters.filter2d(img1 * img2, window)[:, :, pad_topleft:-pad_bottomright,
              pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = torch.ones(sigma12.shape)

    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
            (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

    return torch.mean(qindex_map)



def get_metrics_reduced(img1, img2):
    # input: img1 {the pan-sharpened image}, img2 {the ground-truth image}
    # return: (larger better) psnr, ssim, scc, (smaller better) sam, ergas,uiqi
    m1 = psnr_loss(img1, img2, 1.)
    m2 = ssim(img1, img2, 11, 'mean', 1.)
    m3 = cc(img1, img2)
    m4 = sam(img1, img2)
    m5 = ergas(img1, img2)
    m6 = uiqi(img1, img2)
    return [m1.item(), m2.item(), m3.item(), m4.item(), m5.item(), m6.item()]#item()返回浮点型数据


def read_img(img_dir):
    img = cv2.imread(img_dir)
    img = np.array(img).astype('float32') / 255.
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)#（256，256，3）--》（3，256，256）
    img = np.expand_dims(img,0)
    return torch.Tensor(img)


def QIndex_torch(a, b, eps=1e-8):
    r"""
    Look at paper:
    `A universal image quality index` for details

    Args:
        a (torch.Tensor): one-channel images, shape like [N, H, W]
        b (torch.Tensor): one-channel images, shape like [N, H, W]
    Returns:
        torch.Tensor: Q index value of all images
    """
    E_a = torch.mean(a, dim=(1, 2))
    E_a2 = torch.mean(a * a, dim=(1, 2))
    E_b = torch.mean(b, dim=(1, 2))
    E_b2 = torch.mean(b * b, dim=(1, 2))
    E_ab = torch.mean(a * b, dim=(1, 2))
    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b
    return torch.mean(4 * cov_ab * E_a * E_b / ( (var_a + var_b) * (E_a ** 2 + E_b ** 2) + eps) )


def D_lambda_torch(l_ms, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_lambda value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += torch.abs(QIndex_torch(ps[:, i, :, :], ps[:, j, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_ms[:, j, :, :]))
    return sum


def D_s_torch(l_ms, pan, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        pan (torch.Tensor): PAN images, shape like [N, C, H, W]
        l_pan (torch.Tensor): LR PAN images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_s value of n images
    """
    l_pan = F.interpolate(pan, scale_factor=1 / 4, mode='bicubic', align_corners=False, recompute_scale_factor=True)
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        sum += torch.abs(QIndex_torch(ps[:, i, :, :], pan[:, 0, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))
    return sum

# def QNR(D_lamda, D_s):
#     return abs((1 - D_lamda()) * (1 - D_s()))

if __name__ == '__main__':
    # imgf_dir = 'D:/remote sensing image/model_out/WV/WV2_MDCUN'  #网络输出图像
    imgf_dir = './results'  # 网络输出图像
    gt_dir = '../DMF-Net-main_guo/datasets/QB/test/label'  #Ground Truth
    img_name = [i for i in os.listdir(imgf_dir)]  # 所有image文件名称读取
    ms_dir = '../DMF-Net-main_guo/datasets/QB/test/ms'
    pan_dir = '../DMF-Net-main_guo/datasets/QB/test/pan'


    metrics = torch.zeros(9, len(img_name))#5个评价指标，25个测试集图像
    i = 0
    for name in os.listdir(imgf_dir):
        imgf = os.path.join(imgf_dir, name)
        gt = os.path.join(gt_dir, name)
        ms = os.path.join(ms_dir, name)
        pan = os.path.join(pan_dir, name)

        ms_ = read_img(ms)
        pan_ = read_img(pan)
        imgf_ = read_img(imgf)
        gt_ = read_img(gt)
        metrics[:6, i] = torch.Tensor(get_metrics_reduced(imgf_, gt_))

        metrics[6, i] = torch.Tensor(D_lambda_torch(ms_,imgf_))
        metrics[7, i] = torch.Tensor(D_s_torch(ms_,pan_,imgf_))
        D_lamda = float(metrics[6, i])
        D_s = float(metrics[7, i])
        metrics[8, i] = abs((1 - D_lamda) * (1 - D_s))
        i = i+1
    # Write the metrics

    f = xlwt.Workbook()#python xlwt写入excel创建workbook
    sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)#创建worksheet
    metric_name = ['PSNR','SSIM','CC','SAM','ERGAS','UIQI','Dλ','Ds','QNR']
    for i in range(len(metric_name)):
        sheet1.write(i+1,0,metric_name[i])#第一列写入评价指标名称
    for j in range(len(img_name)):
       sheet1.write(0,j+1,img_name[j])  ##第一行写入文件名称
    for i in range(len(metric_name)):
        for j in range(len(img_name)):
            sheet1.write(i+1,j+1,float(metrics[i,j]))
    sheet1.write(0,len(img_name)+1,'Mean')
    for i in range(len(metric_name)):
        sheet1.write(i+1,len(img_name)+1,float(metrics.mean(1)[i]))
    f.save('QB.xls')