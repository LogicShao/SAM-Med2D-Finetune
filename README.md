# SAM-Med2D Finetune for BraTS 2021

本仓库围绕 `BraTS 2021` 脑肿瘤分割任务，对 `SAM-Med2D` 做微调、整病例推理、3D 后处理与结果展示，当前已经形成一条完整可复用的链路：

```text
训练 / 评估 -> 整病例自动分割 -> 3D 后处理 -> 3D 可视化 -> Web 结果查看
```

仓库支持两条主要工作流：

- 单任务路径：先把原始 BraTS `NIfTI` 数据预处理为 `PNG + JSON`，再用 `train_singletask.py` 训练。
- 多任务路径：直接读取原始 BraTS 病例目录，使用 4 模态输入，预测 `ET / TC / WT`。

当前实现支持 `Adapter` 和 `LoRA` 两种微调方式，并输出日志、模型、训练曲线与整病例级结果。

除训练与评估外，仓库还包含：

- 整病例自动推理入口 `infer_volume.py`
- 3D 后处理与结果整理 `postprocess_3d.py`
- 3D 可视化页面生成 `visualize_case.py`
- 面向演示与结果查看的 `web_demo/`
- 面向结项整理的 `report/` 与 `outputs/` 正式汇总结果

如果你只想快速查看现成病例结果，建议直接启动 `web_demo`；如果你要复现实验或替换模型，再使用训练与整病例推理脚本。

## 0. 结项概览

当前项目在结果口径上已经收敛为两种模式：

- `standard` 标准模式：对应 `Adapter baseline`，强调整体病灶范围分析，只关注 `WT` 与总体肿瘤信息。
- `multiclass` 多类别分析模式：对应 `Adapter + class_boxes_points + current default ET`，用于观察 `WT / TC / ET` 的区域分布和分类体积。

当前可直接作为结项材料引用的结论如下：

1. `Adapter baseline` 是当前正式默认基线。
   - `fixed20`: `post Mean Dice = 0.528955`
   - `confirm_large_unseen`: `post Mean Dice = 0.546043`
2. `Adapter g4` 在机制上仍成立，但角色是“机制验证增强组”，不是默认入口。
   - `fixed20`: 相对 `Adapter baseline` 提升 `+0.007169`
   - `confirm_large_unseen`: 相对 `Adapter baseline` 提升 `+0.001433`
3. 多类别 prompt 主路线已经建立。
   - `fixed20` 上，`Adapter + class_boxes_points` 相对 `Adapter baseline` 的 `post Mean Dice` 提升 `+0.018896`
   - `confirm_large_unseen` 上，相对 `Adapter baseline` 提升 `+0.006567`
4. ET-only 小范围调参没有找到优于当前 `default ET` 的新版本。
   - `recommended_variant = null`
   - `should_advance_to_confirm_large_unseen = false`
5. 历史 `LoRA` 大样本结果继续保留为对照资料，但不再作为当前默认 pipeline 的依据。

## 0.1 重要文档与正式实验索引

结项材料不要直接引用零散病例目录，优先引用下面这些主文档和汇总文件。

| 类型 | 说明 | 主文档 | 对应结果文件 |
| --- | --- | --- | --- |
| 总览入口 | 项目概况、目录、命令、正式实验索引 | [`README.md`](README.md) | - |
| 项目总报告 | 结项叙事、阶段性结论、系统口径说明 | [`report/report.md`](report/report.md) | - |
| 默认基线验证 | `Adapter baseline / g4` 在 `fixed20` 与 `confirm_large_unseen` 上的正式验证 | [`report/adapter_verification.md`](report/adapter_verification.md) | [`outputs/stage7_adapter_verification/summary/adapter_comparison.md`](outputs/stage7_adapter_verification/summary/adapter_comparison.md), [`outputs/stage7_adapter_verification/summary/final_recommendation.md`](outputs/stage7_adapter_verification/summary/final_recommendation.md) |
| 历史大样本确认 | 历史 `LoRA baseline / g4` 在 `confirm_large_unseen` 上的正式确认结果 | [`report/confirm_large_unseen_confirmation.md`](report/confirm_large_unseen_confirmation.md) | [`outputs/stage6_large_confirmation/report/confirm_large_unseen_confirmation_summary.json`](outputs/stage6_large_confirmation/report/confirm_large_unseen_confirmation_summary.json) |
| 多类别 prompt 正式结果 | `class_boxes_points` 在 `fixed20` 与 `confirm_large_unseen` 上的正式汇总 | 本 README 第 7.3 节 | [`outputs/stage8_class_prompt_ablation/fixed20_adapter_class_boxes_points/summary.md`](outputs/stage8_class_prompt_ablation/fixed20_adapter_class_boxes_points/summary.md), [`outputs/stage8_class_prompt_ablation/confirm_large_unseen_adapter_class_boxes_points/summary.md`](outputs/stage8_class_prompt_ablation/confirm_large_unseen_adapter_class_boxes_points/summary.md) |
| ET 调参结论 | ET-only 小范围调参、冻结默认 ET 配置 | [`report/et_prompt_tuning_report.md`](report/et_prompt_tuning_report.md) | [`outputs/stage9_et_prompt_tuning/et_prompt_tuning.md`](outputs/stage9_et_prompt_tuning/et_prompt_tuning.md), [`outputs/stage9_et_prompt_tuning/recommended_et_variant.md`](outputs/stage9_et_prompt_tuning/recommended_et_variant.md) |
| Demo 使用说明 | `web_demo` 的结构、启动方式与展示链路 | [`web_demo/README.md`](web_demo/README.md) | `outputs/web_demo_runs/` |

## 1. 仓库结构

```text
.
├── train_singletask.py          # 单任务训练入口
├── train_multitask.py           # 多任务训练入口
├── evaluate_baseline.py         # 原始 SAM-Med2D 基线评估
├── infer_volume.py              # 整病例自动推理入口
├── postprocess_3d.py            # 3D 后处理与层级约束
├── visualize_case.py            # 3D HTML 可视化生成
├── preprocess_brats.py          # BraTS 预处理脚本
├── create_subset.py             # 抽样小数据集
├── DataLoader.py                # 单任务数据集与 collate
├── multitask_dataset.py         # 多任务 BraTS 数据集
├── metrics.py                   # Dice / IoU 指标
├── utils.py                     # loss、prompt、box、日志工具
├── tools/                       # YOLO 数据准备、训练与辅助脚本
├── web_demo/                    # Web 结果查看与单病例处理入口
├── segment_anything/            # 本地 SAM-Med2D 实现
├── finetune_scripts/            # 固定实验配置脚本
├── pretrain_model/              # 预训练权重
├── data_demo/                   # 预处理后数据示例
└── outputs/                     # 推理、后处理与可视化结果
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

### 2.1 Web Demo 启动

`web_demo` 当前主链路为：

```text
选择病例 / 上传病例 -> 自动分割 -> 3D 重建 -> 结果查看
```

启动方式：

```bash
python -m web_demo.app
```

默认访问地址：

```text
http://127.0.0.1:7860
```

当前 `web_demo` 默认推理配置见 `web_demo/config.py`，包括：

- `finetune_method=adapter`
- `prompt_mode=yolo_box`
- `prompt_box_strategy=top1`
- YOLO checkpoint：`workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt`

当前展示口径同步为：

- 标准模式：`Adapter baseline`
- 多类别分析模式：`Adapter + class_boxes_points + current default ET`

当前样例结果目录为：

- 标准模式样例：`outputs/stage7_adapter_verification/fixed20_adapter_baseline/`
- 多类别分析样例：`outputs/stage9_et_prompt_tuning/fixed20_adapter_class_boxes_points_et_default/`

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

## 7. 正式实验与当前结论

### 7.1 训练阶段结果

以下结果来自当前仓库已有训练日志，便于快速对比不同配置的实际效果。

| 配置 | 最佳验证 Dice | 最佳验证 IoU | 最佳 epoch | 日志位置 |
| --- | --- | --- | --- | --- |
| 单任务 Adapter | 0.8335 | 0.7402 | 5 | `workdir_label_WT/logs/single_task_adapter/metrics.csv` |
| 单任务 LoRA | 0.8113 | 0.6966 | 36 | `workdir_label_WT/logs/single_task_lora/metrics.csv` |
| 多任务 Adapter | 0.7560 | 0.6619 | 24 | `workdir_multi_task/logs/finetune_adapter/metrics.csv` |
| 多任务 LoRA | 0.7265 | 0.6215 | 66 | `workdir_multi_task/logs/finetune_no_stop_lora/metrics.csv` |
| 原始基线 | 0.5303 | 0.3882 | - | `workdir_label_WT/baseline_metrics.json` |

结论：

- 单任务 Adapter 是当前日志里表现最好的配置，收敛也最快。
- 多任务 Adapter 明显优于多任务 LoRA，因此当前整病例默认主模型应以 Adapter 为准。

当前代码中的整病例推理 / `web_demo` 默认主模型为：

- checkpoint：`workdir_multi_task/models/finetune_adapter/best_model.pth`
- `finetune_method`：`adapter`

### 7.2 Stage 7: Adapter 默认基线验证

这是当前“标准模式默认口径”最关键的一组正式实验。

| 数据集 | 配置 | post Mean Dice | ET | TC | WT | 主要结论 |
| --- | --- | --- | --- | --- | --- | --- |
| `fixed20` | Adapter baseline | 0.528955 | 0.374207 | 0.512041 | 0.700616 | 当前标准模式正式基线 |
| `fixed20` | Adapter g4 | 0.536124 | 0.374207 | 0.512041 | 0.722123 | 相对 baseline `+0.007169`，但定位为机制增强组 |
| `confirm_large_unseen` | Adapter baseline | 0.546043 | 0.434018 | 0.556753 | 0.647360 | 当前大样本默认基线 |
| `confirm_large_unseen` | Adapter g4 | 0.547477 | 0.434018 | 0.556753 | 0.651660 | 相对 baseline `+0.001433`，继续保留但不升级为唯一默认 |

对应文档：

- [`report/adapter_verification.md`](report/adapter_verification.md)
- [`outputs/stage7_adapter_verification/summary/adapter_comparison.md`](outputs/stage7_adapter_verification/summary/adapter_comparison.md)
- [`outputs/stage7_adapter_verification/summary/final_recommendation.md`](outputs/stage7_adapter_verification/summary/final_recommendation.md)

### 7.3 Stage 8: 多类别 prompt 正式结果

这组实验决定了当前 `multiclass` 模式的正式口径。

| 数据集 | 配置 | post Mean Dice | ET | TC | WT | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `fixed20` | Adapter baseline | 0.528955 | 0.374207 | 0.512041 | 0.700616 | raw 阶段 `WT = TC = ET` 为 `20/20` |
| `fixed20` | Adapter + class_boxes_points | 0.547851 | 0.404318 | 0.537971 | 0.701264 | 相对 baseline `+0.018896`，raw 三类塌缩降到 `0/20` |
| `confirm_large_unseen` | Adapter baseline | 0.546043 | 0.434018 | 0.556753 | 0.647360 | 当前标准模式基线 |
| `confirm_large_unseen` | Adapter + class_boxes_points | 0.552610 | 0.420123 | 0.587752 | 0.649955 | 相对 baseline `+0.006567`，TC/WT 提升，ET 仍有优化空间 |

结论：

- `class_boxes_points` 已经建立起可用的多类别 prompt 主路线。
- 多类别模式不再依赖历史三类相同掩码结果。
- 当前 `multiclass` 模式应读取 `stage8/9` 这条结果链，而不是复用标准模式结果。

对应结果文件：

- [`outputs/stage8_class_prompt_ablation/fixed20_adapter_baseline/summary.md`](outputs/stage8_class_prompt_ablation/fixed20_adapter_baseline/summary.md)
- [`outputs/stage8_class_prompt_ablation/fixed20_adapter_class_boxes_points/summary.md`](outputs/stage8_class_prompt_ablation/fixed20_adapter_class_boxes_points/summary.md)
- [`outputs/stage8_class_prompt_ablation/confirm_large_unseen_adapter_class_boxes_points/summary.md`](outputs/stage8_class_prompt_ablation/confirm_large_unseen_adapter_class_boxes_points/summary.md)

### 7.4 Stage 9: ET Prompt 调参结论

Stage 9 只围绕 `ET` 提示做低风险小范围收敛实验，不改 `WT/TC` 主逻辑。

| 变体 | post Mean Dice | ET Dice | 结论 |
| --- | --- | --- | --- |
| `default` | 0.547851 | 0.404318 | 当前默认 ET 配置 |
| `q92_pad4_p2_n2` | 0.547799 | 0.401877 | 最接近默认，但仍未超过默认 |
| `q95_pad8_p1_n2_widefb` | 0.545067 | 0.402407 | fallback 过多，raw 层级稳定性下降 |
| `q95_pad4_p1_n2` | 0.532104 | 0.347318 | ET 明显退化 |
| `q90_pad2_p1_n0` | 0.527140 | 0.341197 | 去掉负点后退化更明显 |

结论：

- 本轮没有找到优于 `default` 的 ET-only 版本。
- `recommended_variant = null`
- `should_advance_to_confirm_large_unseen = false`
- 当前多类别分析模式继续采用 `fixed20_adapter_class_boxes_points_et_default`

对应文档：

- [`report/et_prompt_tuning_report.md`](report/et_prompt_tuning_report.md)
- [`outputs/stage9_et_prompt_tuning/et_prompt_tuning.md`](outputs/stage9_et_prompt_tuning/et_prompt_tuning.md)
- [`outputs/stage9_et_prompt_tuning/recommended_et_variant.md`](outputs/stage9_et_prompt_tuning/recommended_et_variant.md)

### 7.5 Stage 6: 历史 confirm_large_unseen 对照

这组结果仍然有参考价值，但只应当作为历史对照，不应覆盖当前 Adapter 默认口径。

| 数据集 | 配置 | post Mean Dice | WT | 结论 |
| --- | --- | --- | --- | --- |
| `confirm_large_unseen` | LoRA baseline | 0.536181 | 0.620401 | 历史主对照 |
| `confirm_large_unseen` | LoRA g4 | 0.535515 | 0.618403 | 相对 baseline `-0.000666`，不支持升级为默认 |

对应文档：

- [`report/confirm_large_unseen_confirmation.md`](report/confirm_large_unseen_confirmation.md)
- [`outputs/stage6_large_confirmation/report/confirm_large_unseen_confirmation_summary.json`](outputs/stage6_large_confirmation/report/confirm_large_unseen_confirmation_summary.json)

## 8. 整病例推理、展示与结项口径

当前仓库已经不再停留在“只会训练”的阶段，而是完成了从原始 BraTS 病例到 3D `NIfTI` 输出、后处理和可视化预览的闭环：

- 已实现整病例推理入口 `infer_volume.py`，可输出 `ET.nii.gz`、`TC.nii.gz`、`WT.nii.gz`、`combined_label.nii.gz`
- 已实现 `postprocess_3d.py`，支持闭运算、开运算、空洞填充、连通域筛选、Z 轴平滑以及 `ET ⊆ TC ⊆ WT` 层级约束
- 已实现 `visualize_case.py`，可生成 Raw / Post 对比的 3D HTML 预览
- 已实现 `web_demo`，可做病例浏览、模式切换、3D 查看与关键切片查看

当前 `web_demo` 结果页重点展示：

- 病例信息与处理状态
- 3D HTML 结果
- 2D 关键切片与叠加图
- 基于现有分割结果与 spacing 的定量分析

体积计算公式为：

```text
volume_ml = voxel_count * spacing_x * spacing_y * spacing_z / 1000
```

展示口径说明：

- 标准模式强调整体病灶范围分析，只显示 `WT / 总体肿瘤`
- 多类别分析模式展示 `WT / TC / ET` 分布与分类体积
- `g4` 适合用于展示 `WT-only continuity` 机制，不应表述为唯一默认配置

## 9. 注意事项

- `train_singletask.py` 和 `train_multitask.py` 使用的数据格式不同，不能混用。
- 多任务训练会自动把图像编码器输入层改成 4 通道，以适配 BraTS 四模态输入。
- `split_raw_data.py` 仍是本地硬编码脚本，不属于当前推荐工作流。
- `web_demo` 的基础定量分析依赖现有 mask 与 spacing；若结果目录缺少对应文件，页面会降级显示为“暂不可用”。
- 如果目标是结项整理，优先引用第 `0.1` 节列出的正式文档，不要直接拼接零散病例目录。
- 仓库当前没有独立自动化测试目录，改动后建议至少运行：

```bash
python -m compileall .
```

再补一次小样本训练或评估命令。

## 10. 许可证

许可证见根目录 [LICENSE](LICENSE)。`segment_anything/` 相关实现保留原始项目的许可证说明。
