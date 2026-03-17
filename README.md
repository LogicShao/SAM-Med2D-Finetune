# SAM-Med2D Finetune for BraTS 2021

本仓库用于将 `SAM-Med2D` 微调到 `BraTS 2021` 脑肿瘤分割任务，围绕实际训练流程提供了两条路径：

- 单任务路径：先将原始 BraTS `NIfTI` 数据预处理为 `PNG + JSON` 索引，再使用 `train_singletask.py` 训练。
- 多任务路径：直接读取原始 BraTS 病例目录，使用 4 模态输入，同时预测 `ET / TC / WT`。

当前实现支持 `Adapter` 和 `LoRA` 两种微调方式，并包含基线评估、小样本抽样、训练日志与曲线导出。

## 1. 仓库结构

```text
.
├── train_singletask.py          # 单任务/单掩码训练入口
├── train_multitask.py           # 多任务 BraTS 训练入口
├── evaluate_baseline.py         # 原始 SAM-Med2D 基线评估
├── preprocess_brats.py          # 将 BraTS 预处理为 2D PNG + JSON
├── create_subset.py             # 从已划分好的 train/val 中抽样小数据集
├── DataLoader.py                # 单任务数据集与 collate 逻辑
├── multitask_dataset.py         # 多任务 BraTS 数据集
├── metrics.py                   # Dice / IoU 指标
├── utils.py                     # loss、prompt、box、日志等工具函数
├── segment_anything/            # 本地 SAM-Med2D 模型实现
├── finetune_scripts/            # 固定实验配置脚本
├── pretrain_model/              # 预训练权重
└── data_demo/                   # 预处理后数据格式示例
```

## 2. 环境准备

建议使用带 CUDA 的 Python 环境。仓库默认训练设备为 `cuda`。

1. 先按你的平台单独安装 `torch` 和 `torchvision`
2. 再安装仓库内其余依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 不再固定绑定某个 CUDA 版本，便于复用你本机已经安装好的 PyTorch 环境。

默认预训练权重路径为：

```text
pretrain_model/sam-med2d_b.pth
```

仓库中已经包含该权重文件，默认模型类型使用 `vit_b`。

## 3. 数据准备

### 3.1 原始数据目录

多任务训练直接读取原始 BraTS 病例目录，建议整理成：

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

`train_multitask.py` 会在训练时直接构造 4 通道输入，并从 `seg` 标签生成 `ET / TC / WT` 三个掩码。

### 3.2 可选：从大数据集中抽样做烟雾测试

如果你已经有一个完整的 `train/val` 根目录，可以先抽样出小规模子集：

```bash
python create_subset.py --source_root data_brats_raw_all --dest_root data_brats_raw --train_num 20 --val_num 4
```

### 3.3 单任务路径的数据预处理

`train_singletask.py` 不直接读取原始 `NIfTI`，而是读取 `preprocess_brats.py` 生成的 `PNG + JSON` 数据集。

示例：

```bash
python preprocess_brats.py \
  --train_data_path data_brats_raw/train \
  --val_data_path data_brats_raw/val \
  --processed_data_path data_brats_WT_TC \
  --labels WT TC
```

生成结果大致如下：

```text
data_brats_WT_TC/
├── images/train/*.png
├── images/val/*.png
├── labels/train/*.png
├── labels/val/*.png
├── image2label_train.json
├── label2image_val.json
└── label2image_train.json / image2label_val.json
```

可以参考 `data_demo/` 查看预处理后数据格式。

## 4. 训练

### 4.1 单任务训练

最小命令：

```bash
python train_singletask.py \
  --finetune_method adapter \
  --data_path data_brats_WT_TC \
  --work_dir workdir_label_WT_TC
```

改用 LoRA：

```bash
python train_singletask.py \
  --finetune_method lora \
  --data_path data_brats_WT_TC \
  --work_dir workdir_label_WT_TC
```

也可以直接运行固定配置脚本：

```bash
python finetune_scripts/single_task/adapter.py
python finetune_scripts/single_task/lora.py
```

### 4.2 多任务训练

最小命令：

```bash
python train_multitask.py \
  --finetune_method lora \
  --train_data_path data_brats_raw/train \
  --val_data_path data_brats_raw/val \
  --work_dir workdir_multi_task
```

改用 Adapter：

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

多任务脚本支持：

- `--train_subset_size` / `--val_subset_size`：快速调试
- `--num_classes`：默认 `3`
- `--input_channels`：默认 `4`

## 5. 基线评估

`evaluate_baseline.py` 用于评估未微调的原始 `SAM-Med2D` 权重在预处理后验证集上的表现：

```bash
python evaluate_baseline.py \
  --data_path data_brats_WT_TC \
  --work_dir workdir_label_WT_TC
```

输出会保存到 `baseline_metrics.json` 和日志文件中。

## 6. 训练输出

训练输出统一保存在 `work_dir` 下：

```text
workdir_xxx/
├── logs/<run_name>/timestamp.log
├── logs/<run_name>/metrics.csv
├── models/<run_name>/
└── plots/<run_name>/
```

常见产物包括：

- `metrics.csv`：每个 epoch 的 loss / Dice / IoU
- `plots/`：训练曲线图
- `models/`：
  - 单任务 Adapter：`best_model.pth`
  - 单任务 LoRA：`lora_encoder_best/`
  - 多任务 Adapter：`best_model.pth`
  - 多任务 LoRA：`lora_adapters/`

## 7. 注意事项

- `train_singletask.py` 和 `train_multitask.py` 使用的数据格式不同，不能混用。
- 多任务训练会自动把图像编码器输入层改成 4 通道，以适配 BraTS 四模态输入。
- `split_raw_data.py` 仍是本地硬编码脚本，不属于当前推荐工作流；如需使用，请先改成你自己的路径。
- 仓库当前没有独立的自动化测试目录。改动后建议至少运行：

```bash
python -m compileall .
```

以及一次小样本训练或评估命令。

## 8. 许可证

许可证见根目录 [LICENSE](LICENSE)。`segment_anything/` 相关实现保留原始项目的许可证说明。
