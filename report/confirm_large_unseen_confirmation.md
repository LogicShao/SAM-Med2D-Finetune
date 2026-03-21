# confirm_large_unseen 大样本确认实验报告

## 1. 数据集与病例数

本次确认实验仅使用更大未参与调参的验证子集 `confirm_large_unseen`，不启动 `full_val_all`。

当前验证集划分如下：

| 集合 | 病例数 | 说明 |
| --- | ---: | --- |
| `full_val_all` | 187 | 当前验证集全部可用病例 |
| `fixed20` | 20 | 既有正式回归固定病例集 |
| `hard8` | 8 | 既有困难病例集 |
| `confirm_large_unseen` | 167 | 用于本轮确认实验的更大未参与调参样本 |

病例列表已落盘，便于复现：

- `outputs/stage6_large_confirmation/case_lists/fixed20.txt`
- `outputs/stage6_large_confirmation/case_lists/hard8.txt`
- `outputs/stage6_large_confirmation/case_lists/confirm_large_unseen_cases.txt`
- `outputs/stage6_large_confirmation/case_lists/full_val_all_cases.txt`

## 2. 比较组配置

本轮只做确认，不调参、不改模型、不改代码逻辑。

| 组别 | 配置说明 | 状态 |
| --- | --- | --- |
| baseline | `conf=0.05, iou=0.60, prompt=top1, z_prompt_mode=none, wt_continuity_enabled=false` | 已完成 |
| g4 | baseline + `WT gate(score=0.08, center=72, area=0.33-3.0, dilate=1, blur=3)` | 已完成 |
| g0 | baseline + `WT gate(score=0.15, center=48, area=0.5-2.0, dilate=1, blur=3)` | 本轮未运行 |

## 3. 实际运行命令

baseline：

```powershell
$cases = Get-Content "D:/proj/SAM-Med2D-Finetune/outputs/stage6_large_confirmation/case_lists/confirm_large_unseen.txt"
& "C:/Users/acs/.conda/envs/Brain-Tumor-Segmentation/python.exe" "D:/proj/SAM-Med2D-Finetune/batch_validate_postprocess.py" `
  --cases_root "D:/proj/SAM-Med2D-Finetune/data_brats_raw/val" `
  --output_root "D:/proj/SAM-Med2D-Finetune/outputs/stage6_large_confirmation/runs/confirm_large_unseen_baseline" `
  --case_ids $cases `
  --sam_checkpoint "D:/proj/SAM-Med2D-Finetune/pretrain_model/sam-med2d_b.pth" `
  --finetuned_checkpoint "D:/proj/SAM-Med2D-Finetune/workdir_multi_task/models/finetune_no_stop_lora/lora_adapters" `
  --finetune_method lora --prompt_mode yolo_box --image_size 256 --input_channels 4 `
  --encoder_adapter true --device cuda --threshold 0.5 --use_amp true `
  --yolo_checkpoint "D:/proj/SAM-Med2D-Finetune/workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt" `
  --yolo_conf 0.05 --yolo_iou 0.60 --yolo_max_det 2 --yolo_topk 2 `
  --prompt_box_strategy top1 --z_prompt_mode none --wt_continuity_enabled false `
  --postprocess true --closing_radius 2 --opening_radius 1 `
  --wt_keep_largest true --keep_topk_tc 1 --keep_topk_et 1 --z_smooth_iterations 3
```

g4：

```powershell
$cases = Get-Content "D:/proj/SAM-Med2D-Finetune/outputs/stage6_large_confirmation/case_lists/confirm_large_unseen_cases.txt"
& "C:/Users/acs/.conda/envs/Brain-Tumor-Segmentation/python.exe" "D:/proj/SAM-Med2D-Finetune/batch_validate_postprocess.py" `
  --cases_root "D:/proj/SAM-Med2D-Finetune/data_brats_raw/val" `
  --output_root "D:/proj/SAM-Med2D-Finetune/outputs/stage6_large_confirmation/runs/confirm_large_unseen_g4_formal" `
  --case_ids $cases `
  --sam_checkpoint "D:/proj/SAM-Med2D-Finetune/pretrain_model/sam-med2d_b.pth" `
  --finetuned_checkpoint "D:/proj/SAM-Med2D-Finetune/workdir_multi_task/models/finetune_no_stop_lora/lora_adapters" `
  --finetune_method lora --prompt_mode yolo_box --image_size 256 --input_channels 4 `
  --encoder_adapter true --device cuda --threshold 0.5 --use_amp true `
  --yolo_checkpoint "D:/proj/SAM-Med2D-Finetune/workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt" `
  --yolo_conf 0.05 --yolo_iou 0.60 --yolo_max_det 2 --yolo_topk 2 `
  --prompt_box_strategy top1 --z_prompt_mode none --wt_continuity_enabled true `
  --wt_continuity_score_thresh 0.08 --wt_continuity_center_shift_max 72.0 `
  --wt_continuity_area_ratio_min 0.33 --wt_continuity_area_ratio_max 3.0 `
  --wt_continuity_mask_dilate_iters 1 --wt_continuity_mask_blur_kernel 3 `
  --postprocess true --closing_radius 2 --opening_radius 1 `
  --wt_keep_largest true --keep_topk_tc 1 --keep_topk_et 1 --z_smooth_iterations 3
```

## 4. 主要结果表

### 4.1 汇总指标

| 组别 | Post Mean Dice | ET | TC | WT | 相对 baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.536181 | 0.432519 | 0.555623 | 0.620401 | 0.000000 |
| g4 | 0.535515 | 0.432519 | 0.555623 | 0.618403 | -0.000666 |

### 4.2 g4 相对 baseline 的类别变化

| 指标 | Delta |
| --- | ---: |
| Post Mean Dice | -0.000666 |
| ET post Dice | 0.000000 |
| TC post Dice | 0.000000 |
| WT post Dice | -0.001998 |

## 5. 关键统计摘要

### 5.1 case-level 统计

| 统计项 | 数值 |
| --- | ---: |
| win | 86 |
| tie | 13 |
| loss | 68 |
| mean delta | -0.000666 |
| median delta | 0.000012 |
| bootstrap 95% CI | [-0.005824, 0.004504] |

补充说明：

- `median delta` 仍接近 0，说明多数病例的变化幅度并不大。
- `bootstrap 95% CI` 跨过 0，说明在当前更大未见样本上，`g4` 相对 baseline 的整体收益并不稳健。

### 5.2 WT continuity 统计

baseline 未启用 `WT continuity`，对应统计均为 0。  
g4 的统计如下：

| 统计项 | 数值 |
| --- | ---: |
| eligible_total | 18790 |
| trigger_total | 9848 |
| rescue | 1146 |
| neutral | 2353 |
| harm | 6349 |

`trigger_reasons` 分布：

| 原因 | 次数 |
| --- | ---: |
| missing_box | 7690 |
| low_score | 1340 |
| center_jump | 531 |
| area_jump | 1006 |

### 5.3 耗时与结果生成

耗时来自结果目录创建时间到 `summary_metrics.json` 修改时间的粗略估算，不能视为严格 profiling。

| 组别 | 总耗时估算（秒） | 平均每例（秒） | 3D 结果生成成功率 |
| --- | ---: | ---: | ---: |
| baseline | 1758.176 | 10.528 | 167/167 = 100% |
| g4 | 1964.373 | 11.763 | 167/167 = 100% |

## 6. 结果路径

- baseline 汇总：`outputs/stage6_large_confirmation/runs/confirm_large_unseen_baseline/summary.md`
- baseline 指标：`outputs/stage6_large_confirmation/runs/confirm_large_unseen_baseline/summary_metrics.json`
- g4 汇总：`outputs/stage6_large_confirmation/runs/confirm_large_unseen_g4_formal/summary.md`
- g4 指标：`outputs/stage6_large_confirmation/runs/confirm_large_unseen_g4_formal/summary_metrics.json`
- 对比摘要 JSON：`outputs/stage6_large_confirmation/report/confirm_large_unseen_confirmation_summary.json`

## 7. 最终结论

本轮 `confirm_large_unseen` 大样本确认实验不支持“`g4` 在更大未参与调参样本上优于 baseline”这一判断。

可以冻结的结论如下：

1. `g4` 在 `confirm_large_unseen` 167 例上的 `post Mean Dice` 为 `0.535515`，低于 baseline 的 `0.536181`，差值为 `-0.000666`。
2. `ET/TC` 与 baseline 完全一致，没有退化，但也没有增益。
3. `WT` 仍是唯一受影响的类别，但本轮并未延续 fixed20 上的正收益，`WT post Dice` 从 `0.620401` 下降到 `0.618403`。
4. `WT continuity` 的触发仍以 `missing_box` 为主，但在更大未见样本上 `harm` 仍明显偏高，达到 `6349`。
5. fixed20 与 `confirm_large_unseen` 的结论不完全一致，因此当前不能再把 `g4` 表述为“更大样本默认最优配置”。

基于现有证据，结项阶段更稳妥的表述应为：

- baseline 仍是更稳妥的主对照与默认配置；
- g4 保留为 `WT-only continuity` 已完成机制验证的代表配置，用于展示“WT missing_box 补救”的思路与 fixed20 上的正收益；
- `full_val_all` 暂未运行，待本轮结果完成归档后再决定是否继续。
