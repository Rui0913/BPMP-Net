# BPMP-Net 中文使用说明

本项目整理自 **prj3113/Net2.py + main3.py**，对应论文
《A Progressive Multiscale Network for Pansharpening with Bidirectional Pyramids》。
原始代码逐字节保存在 `legacy/prj3113/`，来源及 SHA256 见
`docs/source_manifest.json`。完整参数说明见 [英文 README](README.md)。

## 文件对应关系

| 原文件/类 | 整理后的文件/类 |
|---|---|
| `Net2.py / Net` | `bpmp/models/network.py / BPMPNet` |
| `Net2.py / block1` | `network.py / MCAB` |
| `SCExt.py / PanExt` | `skafe.py / SKAFE` |
| `MSA.py / SCAttention` | `scwab.py / SCWAB` |
| `CAB.py` | `window_attention.py`，窗口注意力公共实现 |
| `base_net.py` | `common.py`，卷积与残差基础模块 |
| `main3.py` | `train.py`、`bpmp/engine.py`、`bpmp/losses.py` |
| `dataset.py` | `bpmp/data.py` |
| 原测试及指标脚本 | `predict.py`、`bpmp/metrics.py` |

## 在服务器上运行

先使用适合服务器 CUDA 环境的 PyTorch，再安装 `requirements.txt` 中的依赖。
数据文件夹分别包含 `pan/`、`ms/`、`label/`；同一场景的文件名主体必须相同。
PAN 为 256×256 灰度图，MS 为 64×64 三通道图，GT 为 256×256 三通道图，均为 8 位。
保持 OpenCV 的 **BGR** 通道顺序，归一化到 `[-1,1]`，不会自动调整图像大小。

```bash
python train.py \
  --train-dir /data/QB/train --val-dir /data/QB/val \
  --output-dir runs/qb_legacy --device cuda:0

python predict.py \
  --data-dir /data/QB/test --checkpoint runs/qb_legacy/best.pth \
  --output-dir results/qb --device cuda:0 --evaluate
```

默认保留原实验的学习率 0.001、Adam、DataLoader batch 16、11000 次迭代、
每 50 次验证，以及两路输出各自 `0.6 L1 + 0.3 MSE + 0.1 (1 - MS-SSIM)`。

原训练程序每个“epoch”仅取一个随机批次，并逐张图像更新参数。因此默认
`--epoch-mode legacy --micro-batch-size 1` 保留这种采样和更新方式。
选择 `--epoch-mode full` 才会遍历完整数据集；这会大幅改变训练总量。
选择 `--micro-batch-size 16` 会改变每次更新的样本数和 BatchNorm 行为。
这些选项应作为新的服务器实验记录，不能直接与原来的 11000 次迭代等同。

训练输出包含配置、CSV 日志以及可移植的 `best.pth`、`latest.pth`。
断点续训使用 `--resume runs/qb_legacy/latest.pth`；只加载权重开始新实验使用
`--pretrained /path/to/checkpoint.pth`。每次新实验使用独立输出文件夹。

## 已明确的论文与代码差异

1. 原代码两路输出实际上都是 256×256，第一路在上采样之后输出；论文写的是
   128×128 与 256×256 分别监督。本次保留原模型位置，不擅自增加新输出尺度。
2. 原代码调用 `ms_ssim`，不是单尺度 `ssim`，且 `data_range=1.0`，输入为 `[-1,1]`。
   默认保留这套目标函数。若研究单尺度 SSIM，可显式使用
   `--structure ssim --data-range 2.0`，并重新验证结果。
3. 模型中部分已注册分支未直接贡献最终输出；本次保留其运算、参数名和
   BatchNorm 状态更新，以便严格加载历史双输出模型的 state dict。
4. 整理后的指标采用归一化的 D_lambda/D_s 和标准形式的 QNR，不沿用旧脚本中
   未归一化的求和及绝对值。因此新指标不能视为已复现论文表格。

训练流程已修复数据随机错配、尾批次处理、随机种子被覆盖、验证后未恢复
训练模式等问题。重训练结果可能因此变化，完整说明见
[修改记录](docs/REFACTOR_NOTES.md)。

## 检查与旧权重迁移

```bash
# 不依赖 PyTorch 的静态检查
python scripts/check_source.py

# 下面的数值检查应在服务器安装依赖后运行
python -m unittest discover -s tests -v
python -m pip install -r requirements-legacy.txt
python scripts/verify_model.py --device cuda:0 --backward

# 只对你自己保存、可信的旧整模型权重执行
python scripts/convert_checkpoint.py \
  --input /path/to/model_best_epoch.pth --output converted/best.pth --allow-pickle
```

本地检查与服务器实际训练、推理、性能验证分别记录，见
[验证记录](docs/VALIDATION.md)。仓库不包含数据集、权重或未经验证的实验指标。
