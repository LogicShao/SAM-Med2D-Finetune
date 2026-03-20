# SAM-Med2D Finetune for BraTS 2021

本仓库用于将 `SAM-Med2D` 微调到 `BraTS 2021` 脑肿瘤分割任务，支持两条工作流：

- 单任务路径：先把原始 BraTS `NIfTI` 数据预处理为 `PNG + JSON`，再用 `train_singletask.py` 训练。
- 多任务路径：直接读取原始 BraTS 病例目录，使用 4 模态输入，预测 `ET / TC / WT`。

当前实现支持 `Adapter` 和 `LoRA` 两种微调方式，并输出日志、模型和训练曲线。

## 1. 仓库结构

```text
.
├── train_singletask.py          # 单任务训练入口
├── train_multitask.py           # 多任务训练入口
├── evaluate_baseline.py         # 原始 SAM-Med2D 基线评估
├── preprocess_brats.py          # BraTS 预处理脚本
├── create_subset.py             # 抽样小数据集
├── DataLoader.py                # 单任务数据集与 collate
├── multitask_dataset.py         # 多任务 BraTS 数据集
├── metrics.py                   # Dice / IoU 指标
├── utils.py                     # loss、prompt、box、日志工具
├── segment_anything/            # 本地 SAM-Med2D 实现
├── finetune_scripts/            # 固定实验配置脚本
├── pretrain_model/              # 预训练权重
└── data_demo/                   # 预处理后数据示例
```

## 2. 环境准备

建议使用带 CUDA 的 Python 环境，训练默认设备为 `cuda`。

1. 先按本机平台安装 `torch` 和 `torchvision`
2. 再安装其余依赖：

```bash
pip install -r requirements.txt
```

默认预训练权重：

```text
pretrain_model/sam-med2d_b.pth
```

## 3. 数据准备

### 3.1 原始数据目录

多任务训练直接读取原始 BraTS 病例目录，建议组织为：

```text
data_brats_raw/
├── train/
│   ├── BraTS2021_xxxxx/
│   │   ├── BraTS2021_xxxxx_t1.nii.gz
│   │   ├── BraTS2021_xxxxx_t1ce.nii.gz
│   │   ├── BraTS2021_xxxxx_t2.nii.gz
│   │   ├── BraTS2021_xxxxx_flair.nii.gz
│   │   └── BraTS2021_xxxxx_seg.nii.gz
└── val/
```

`train_multitask.py` 会在训练时构造 4 通道输入，并从 `seg` 生成 `ET / TC / WT` 掩码。

### 3.2 抽样小数据集

如果你已有完整的 `train/val` 目录，可以先抽样做快速验证：

```bash
python create_subset.py --source_root data_brats_raw_all --dest_root data_brats_raw --train_num 20 --val_num 4
```

### 3.3 单任务预处理

`train_singletask.py` 读取的是 `preprocess_brats.py` 生成的 `PNG + JSON` 数据集。

```bash
python preprocess_brats.py \
  --train_data_path data_brats_raw/train \
  --val_data_path data_brats_raw/val \
  --processed_data_path data_brats_WT_TC \
  --labels WT TC
```

生成结构大致如下：

```text
data_brats_WT_TC/
├── images/train/*.png
├── images/val/*.png
├── labels/train/*.png
├── labels/val/*.png
├── image2label_train.json
└── label2image_val.json
```

可以参考 `data_demo/` 查看格式示例。

## 4. 训练

### 4.1 单任务训练

```bash
python train_singletask.py \
  --finetune_method adapter \
  --data_path data_brats_WT_TC \
  --work_dir workdir_label_WT
```

LoRA 版本：

```bash
python train_singletask.py \
  --finetune_method lora \
  --data_path data_brats_WT_TC \
  --work_dir workdir_label_WT
```

固定配置脚本：

```bash
python finetune_scripts/single_task/adapter.py
python finetune_scripts/single_task/lora.py
```

### 4.2 多任务训练

```bash
python train_multitask.py \
  --finetune_method lora \
  --train_data_path data_brats_raw/train \
  --val_data_path data_brats_raw/val \
  --work_dir workdir_multi_task
```

Adapter 版本：

```bash
python train_multitask.py \
  --finetune_method adapter \
  --train_data_path data_brats_raw/train \
  --val_data_path data_brats_raw/val \
  --work_dir workdir_multi_task
```

固定配置脚本：

```bash
python finetune_scripts/multi_task/adapter.py
python finetune_scripts/multi_task/lora.py
```

## 5. 基线评估

```bash
python evaluate_baseline.py \
  --data_path data_brats_WT_TC \
  --work_dir workdir_label_WT
```

输出会保存到 `baseline_metrics.json` 和日志文件。

## 6. 训练输出

训练结果统一保存在 `work_dir` 下：

```text
workdir_xxx/
├── logs/<run_name>/timestamp.log
├── logs/<run_name>/metrics.csv
├── models/<run_name>/
└── plots/<run_name>/
```

常见产物：

- `metrics.csv`：每个 epoch 的 loss / Dice / IoU
- `plots/`：训练曲线图
- `models/`：
  - 单任务 Adapter：`best_model.pth`
  - 单任务 LoRA：`lora_encoder_best/`
  - 多任务 Adapter：`best_model.pth`
  - 多任务 LoRA：`lora_adapters/`

## 7. 训练结果

以下结果来自当前仓库已有日志，便于快速对比不同配置的实际效果。

| 配置 | 最佳验证 Dice | 最佳验证 IoU | 最佳 epoch | 日志位置 |
| --- | --- | --- | --- | --- |
| 单任务 Adapter | 0.8335 | 0.7402 | 5 | `workdir_label_WT/logs/single_task_adapter/metrics.csv` |
| 单任务 LoRA | 0.8113 | 0.6966 | 36 | `workdir_label_WT/logs/single_task_lora/metrics.csv` |
| 多任务 LoRA | 0.7265 | 0.6215 | 66 | `workdir_multi_task/logs/finetune_no_stop_lora/metrics.csv` |
| 原始基线 | 0.5303 | 0.3882 | - | `workdir_label_WT/baseline_metrics.json` |

结论：

- 单任务 Adapter 是当前日志里表现最好的配置，收敛也最快。
- 单任务 LoRA 明显优于基线，但峰值略低于 Adapter。
- 多任务 LoRA 的指标低于单任务结果，符合任务更复杂、输出类别更多的预期。

## 8. 整病例推理与端到端进展

当前仓库已经不再停留在“只会训练”的阶段，而是完成了从原始 BraTS 病例到 3D `NIfTI` 输出、后处理和可视化预览的闭环：

- 已新增整病例推理入口 `infer_volume.py`，支持对单个 BraTS 病例输出 `ET.nii.gz`、`TC.nii.gz`、`WT.nii.gz`、`combined_label.nii.gz`。
- 已新增 3D 后处理模块 `postprocess_3d.py`，支持闭运算、开运算、空洞填充、连通域筛选、Z 轴平滑以及 `ET ⊆ TC ⊆ WT` 层级约束。
- 已新增 `visualize_case.py`，可生成 Raw / Post 对比的 3D HTML 预览。

### 8.1 Upper Bound 提示基线

在 4 个固定验证病例上，使用真值紧致框作为 prompt 的 `upper_bound` 模式，并配合强后处理，得到：

| 设置 | Mean Dice | Mean IoU | 说明 |
| --- | --- | --- | --- |
| `full_image_box` + 后处理 | 0.1807 | 0.1026 | 大量健康层假阳性 |
| `upper_bound` + 后处理 | 0.6787 | 0.5370 | 证明模型本体具备可用上限 |

这说明当前多任务 SAM-Med2D 的主要瓶颈并不完全在分割头，而在提示质量。

### 8.2 YOLO 检测与端到端闭环

仓库现已支持将 BraTS 原始病例切片转换为 YOLO 数据集，并训练检测器为 SAM 提供自动 bbox prompt。当前最优候选检测器为 `workdir_yolo/brats_yolo_dev_img320_v8m`：

- `img320_v8m` 在 Dev 集上达到 `mAP50 = 0.8415`、`mAP50-95 = 0.6312`、`recall = 0.7778`
- 阈值扫描表明 `conf = 0.05` 时 `slice_recall_any_box = 0.9367`，适合作为“宁可多给框，也尽量不漏层”的工作点

基于该 YOLO 模型，`yolo_box -> SAM -> 3D postprocess` 的 4 病例端到端验证结果为：

| 配置 | Raw Mean Dice | Post Mean Dice | Raw Mean IoU | Post Mean IoU |
| --- | --- | --- | --- | --- |
| `yolo_box` + 强后处理 | 0.5106 | 0.5431 | 0.3616 | 0.3984 |

### 8.3 阶段一结论：YOLO 工作点扩展验证

为避免基于 4 病例样本过早下结论，额外在 `data_brats_raw/val` 的固定 20 病例子集上做了阶段一扩展验证，并从基线结果中冻结了 8 个困难病例。

先在 YOLO 切片级召回上扫描 `conf ∈ {0.03, 0.05, 0.08, 0.10, 0.15}`、`iou ∈ {0.50, 0.60, 0.70}`。检测层最激进的候选是 `conf=0.03`，其 `slice_recall_any_box` 更高，但在端到端验证中没有转化为稳定收益。

20 病例端到端结果如下：

| 配置 | Post Mean Dice | Post Dice ET | Post Dice TC | Post Dice WT |
| --- | --- | --- | --- | --- |
| `conf=0.05, iou=0.60` | 0.5240 | 0.3729 | 0.5154 | 0.6839 |
| `conf=0.03, iou=0.60` | 0.5236 | 0.3699 | 0.5137 | 0.6874 |

冻结的 8 个困难病例上，`conf=0.03, iou=0.60` 的 `post Mean Dice` 仅从 `0.3653` 提升到 `0.3656`，增幅极小，且多数病例并未明显改善。

结论：

- 当前阶段一不支持将默认工作点从 `conf=0.05, iou=0.60` 切换到更激进的 detector 阈值。
- 降低 `conf` 可以提升切片级召回，但在当前 `top-1 bbox` prompt 流程下，没有带来稳定的端到端收益。
- 因此更合理的下一步不是继续深挖 detector 阈值，而是进入第二阶段，验证 `top-1` 与 `top-2 + 规则过滤` 的 prompt 策略差异。

结论：项目已经取得实质性进展，当前具备“整病例自动提示分割 + 3D 后处理 + 3D 预览”的完整实验链路；在扩展样本验证后，当前默认 detector 工作点仍建议保持 `conf=0.05, iou=0.60`，后续优化重点应转向 prompt 策略而非继续单独微调 YOLO 阈值。

## 9. 注意事项

- `train_singletask.py` 和 `train_multitask.py` 使用的数据格式不同，不能混用。
- 多任务训练会自动把图像编码器输入层改成 4 通道，以适配 BraTS 四模态输入。
- `split_raw_data.py` 仍是本地硬编码脚本，不属于当前推荐工作流。
- 仓库当前没有独立自动化测试目录，改动后建议至少运行：

```bash
python -m compileall .
```

再补一次小样本训练或评估命令。

## 10. 许可证

许可证见根目录 [LICENSE](LICENSE)。`segment_anything/` 相关实现保留原始项目的许可证说明。
