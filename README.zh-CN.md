# BPMP-Net

**用于全色锐化的双向渐进式多尺度金字塔融合网络**

## 论文信息

- **论文题目：** A progressive multiscale network for pansharpening with bidirectional pyramids
- **作者：** Rui Hao、Xin Jin、Shengfa Miao、Puming Wang、Hongyue Huang、Qian Jiang
- **期刊：** Engineering Applications of Artificial Intelligence，182 (2026)，116059
- **DOI：** `10.1016/j.engappai.2026.116059`
- **论文链接：** [通过 DOI 阅读论文](https://doi.org/10.1016/j.engappai.2026.116059)
- **English：** [English README](README.md)

本项目将高分辨率全色图像（PAN）与低分辨率多光谱图像（MS）融合，生成兼顾
空间细节和光谱信息的高分辨率多光谱图像（HRMS）。论文在 Maryland、QuickBird
和 WorldView-2 数据集上进行了实验。

当前代码整理自 **prj3113/Net2.py + main3.py**。原始代码逐字节保存在
`legacy/prj3113/`，来源和 SHA256 见 [来源记录](docs/source_manifest.json)。
整理版保留历史模型的运算和参数键；实际论文指标复现仍需在服务器上验证。

## 论文创新点

1. **双向渐进式多尺度金字塔融合结构。** 建立 PAN、MS 和融合三个分支，
   在内部 128×128 和 256×256 两个尺度进行跨模态、跨尺度特征交互。通过双向
   信息流结合低尺度的结构与上下文信息，以及原尺度的高频细节，逐步融合空间和光谱信息。
2. **MCAB：多尺度交叉注意力模块。** 利用交叉注意力建立 PAN 与 MS 的相关性，
   配合串行与并行多尺度卷积，结合较大范围的上下文和局部细节，引导空间—光谱融合。
3. **SKAFE：混洗选择性核注意力特征提取模块。** 结合自注意力与选择性核注意力
   提取多尺度特征，利用特征混洗促进信息交换，并通过渐进恢复策略同时保留宏观结构
   和微观细节，减少源图像结构在特征提取中的损失。
4. **SCWAB：空间—通道窗口自注意力模块。** 将空间注意力、通道注意力、普通／
   移位窗口注意力与密集连接结合，建模空间和通道相关性；局部窗口相较全局自注意力
   降低计算开销，通道重标定进一步细化光谱表征。

上述内容概括论文的设计。历史代码的输出尺度和损失实现差异见下文。

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

建议使用 Python 3.10 或更高版本，先准备适合服务器 CUDA／驱动环境的 PyTorch。
克隆仓库并安装其余依赖：

```bash
git clone https://github.com/Rui0913/BPMP-Net.git
cd BPMP-Net
# 当前 PR 合并前，切换到包含整理版代码的分支；合并后可直接使用 main。
git checkout refactor/prj3113-readable
python -m pip install -r requirements.txt
```

`requirements.txt` 给出当前代码依赖的最低版本，不是原实验的环境锁定文件。
请在服务器实验中记录实际包版本。仓库未附带数据集或预训练权重。

### 数据准备

```text
dataset/
  train/
    pan/sample001.png    # 256×256，单通道
    ms/sample001.png     # 64×64，三通道
    label/sample001.png  # 256×256，三通道 GT
  val/
    pan/  ms/  label/
  test/
    pan/  ms/  label/     # 仅导出预测时可以不提供 label
```

数据文件夹分别包含 `pan/`、`ms/`、`label/`；同一场景的文件名主体必须相同。
PAN 为 256×256 灰度图，MS 为 64×64 三通道图，GT 为 256×256 三通道图，均为 8 位。
保持 OpenCV 的 **BGR** 通道顺序，归一化到 `[-1,1]`，不会自动调整图像大小。

### 训练与推理

所有命令在仓库根目录运行，`/data/QB/` 替换为服务器上的实际数据路径。

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

训练输出包含 `config.json`、`history.csv` 以及可移植的 `best.pth`、`latest.pth`。
`best.pth` 按验证损失选择，`latest.pth` 在验证时及最后一次迭代保存。
推理导出第二路／最终输出，将结果裁剪到 `[-1,1]` 后映射到 8 位 PNG。
`--evaluate` 需要 `label/`，并输出逐图及平均指标 `metrics.csv`；没有 GT 时去掉该参数。
当前数据管线使用三通道 8 位图像，不能直接输入原始四波段 QuickBird 或八波段 WorldView-2 产品。
断点续训示例：

```bash
python train.py \
  --train-dir /data/QB/train --val-dir /data/QB/val \
  --output-dir runs/qb_legacy --device cuda:0 \
  --resume runs/qb_legacy/latest.pth
```

只加载权重开始新实验使用 `--pretrained /path/to/checkpoint.pth`。
每次新实验使用独立输出文件夹；`--epochs` 表示目标总迭代／epoch 数。
全部可选参数可通过 `python train.py --help` 和 `python predict.py --help` 查看。

## 论文损失函数

论文第 3.6 节在两个尺度使用相同的复合监督：

```text
L_s = 0.6 * L1(y_s, prediction_s)
    + 0.3 * MSE(y_s, prediction_s)
    + 0.1 * (1 - SSIM(y_s, prediction_s))
L_total = sum(L_s for s in {128, 256})
```

L1 主要约束像素辐射与光谱一致性，MSE 加强对较大重建误差的惩罚，
SSIM 约束边缘和纹理结构。代码默认实际使用 MS-SSIM，详见下面的差异说明。

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

## 论文引用

如果本项目对你的研究有帮助，请引用：

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
