# Adapter 版本补充验证报告

## 1. 实验目的

当前仓库历史正式回归、`confirm_large_unseen` 确认实验与早期 `web demo` 默认配置，实际沿用的是多任务 `LoRA` 主模型口径；但训练日志与仓库既有结论已经表明，多任务 `Adapter` 的模型能力优于多任务 `LoRA`。因此，本轮需要在不改算法、不改调参口径的前提下，补齐多任务 `Adapter` 在 `fixed20` 与 `confirm_large_unseen` 上的正式 baseline / g4 证据链，用于统一报告结论与默认模型决策依据。

## 2. 实验设置

本轮补充验证共覆盖 4 组实验：

- `fixed20` Adapter baseline
- `fixed20` Adapter g4
- `confirm_large_unseen` Adapter baseline
- `confirm_large_unseen` Adapter g4

约束如下：

- 除 `finetune_method=adapter` 与 `finetuned_checkpoint=workdir_multi_task/models/finetune_adapter/best_model.pth` 外，其余变量全部冻结
- 不做新参数搜索
- 不改算法逻辑
- 结果格式保持与既有 LoRA 评估一致，便于横向比较

正式结果目录如下：

- `outputs/stage7_adapter_verification/fixed20_adapter_baseline/`
- `outputs/stage7_adapter_verification/fixed20_adapter_g4/`
- `outputs/stage7_adapter_verification/confirm_large_unseen_adapter_baseline/`
- `outputs/stage7_adapter_verification/confirm_large_unseen_adapter_g4/`

汇总结果见：

- `outputs/stage7_adapter_verification/summary/adapter_comparison.json`
- `outputs/stage7_adapter_verification/summary/adapter_comparison.md`
- `outputs/stage7_adapter_verification/summary/final_recommendation.md`

## 3. 运行配置

固定配置如下：

- Adapter checkpoint：`workdir_multi_task/models/finetune_adapter/best_model.pth`
- detector：`conf=0.05, iou=0.60`
- prompt policy：`top1`
- `z_prompt_mode`：`none`
- `image_size=256`
- `input_channels=4`
- `threshold=0.5`
- 后处理：`closing_radius=2, opening_radius=1, wt_keep_largest=true, keep_topk_tc=1, keep_topk_et=1, z_smooth_iterations=3`

`g4` 仅在 baseline 基础上额外启用以下固定参数：

- `wt_continuity_enabled=true`
- `wt_continuity_score_thresh=0.08`
- `wt_continuity_center_shift_max=72.0`
- `wt_continuity_area_ratio_min=0.33`
- `wt_continuity_area_ratio_max=3.0`
- `wt_continuity_mask_dilate_iters=1`
- `wt_continuity_mask_blur_kernel=3`

## 4. 结果

### 4.1 聚合指标

| 数据集 | 模型 | 组别 | overall post Mean Dice | ET post Dice | TC post Dice | WT post Dice | 总耗时估算(s) | 平均每例(s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed20` | Adapter | baseline | `0.528955` | `0.374207` | `0.512041` | `0.700616` | `271.641` | `13.582` |
| `fixed20` | Adapter | g4 | `0.536124` | `0.374207` | `0.512041` | `0.722123` | `327.881` | `16.394` |
| `fixed20` | LoRA | baseline | `0.524049` | `0.372901` | `0.515366` | `0.683878` | `250.512` | `12.526` |
| `fixed20` | LoRA | g4 | `0.528580` | `0.372901` | `0.515366` | `0.697473` | `343.868` | `17.193` |
| `confirm_large_unseen` | Adapter | baseline | `0.546043` | `0.434018` | `0.556753` | `0.647360` | `2130.062` | `12.755` |
| `confirm_large_unseen` | Adapter | g4 | `0.547477` | `0.434018` | `0.556753` | `0.651660` | `2552.847` | `15.287` |
| `confirm_large_unseen` | LoRA | baseline | `0.536181` | `0.432519` | `0.555623` | `0.620401` | `1758.176` | `10.528` |
| `confirm_large_unseen` | LoRA | g4 | `0.535515` | `0.432519` | `0.555623` | `0.618403` | `1964.373` | `11.763` |

### 4.2 Adapter 内部 g4 vs baseline

| 数据集 | overall delta | ET delta | TC delta | WT delta | case-level win / tie / loss |
| --- | ---: | ---: | ---: | ---: | --- |
| `fixed20` | `+0.007169` | `+0.000000` | `+0.000000` | `+0.021507` | `14 / 0 / 6` |
| `confirm_large_unseen` | `+0.001433` | `+0.000000` | `+0.000000` | `+0.004300` | `86 / 7 / 74` |

补充说明：

- `fixed20` 上，Adapter g4 的整体增益高于历史 LoRA g4，且 `WT` 增益更明显
- `confirm_large_unseen` 上，Adapter g4 仍保持正增益，而历史 LoRA g4 在同集上为 `-0.000666`

### 4.3 Adapter vs LoRA

| 数据集 | 组别 | overall delta | ET delta | TC delta | WT delta |
| --- | --- | ---: | ---: | ---: | ---: |
| `fixed20` | baseline | `+0.004906` | `+0.001306` | `-0.003325` | `+0.016737` |
| `fixed20` | g4 | `+0.007544` | `+0.001306` | `-0.003325` | `+0.024650` |
| `confirm_large_unseen` | baseline | `+0.009862` | `+0.001499` | `+0.001130` | `+0.026959` |
| `confirm_large_unseen` | g4 | `+0.011962` | `+0.001499` | `+0.001130` | `+0.033257` |

### 4.4 WT continuity 统计（Adapter g4）

| 数据集 | eligible_total | trigger_total | rescue | neutral | harm | trigger_reasons |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fixed20` | `2621` | `1349` | `162` | `305` | `882` | `{"missing_box": 1036, "low_score": 193, "center_jump": 79, "area_jump": 151}` |
| `confirm_large_unseen` | `21027` | `11969` | `1249` | `2592` | `8128` | `{"low_score": 1528, "missing_box": 9518, "center_jump": 621, "area_jump": 1135}` |

## 5. 结论

本轮 Adapter 补证结果支持以下结论：

1. 默认 baseline 应切换到 Adapter。  
在 `fixed20` 与 `confirm_large_unseen` 两个正式数据集上，Adapter baseline 相对 LoRA baseline 的 `post Mean Dice` 分别提升 `+0.004906` 与 `+0.009862`，`WT` 也同步提升。

2. g4 仍应保留，但更适合作为稳定的机制验证组。  
本轮 Adapter g4 在两个数据集上都取得正增益，说明 `WT-only continuity` 在 Adapter 主模型上依然有效；但从配置角色划分上，baseline 仍应保留“正式主对照与默认入口”的身份，g4 更适合作为可解释的机制增强组。

3. web demo 默认模型应同步切换到 Adapter baseline。  
既然正式默认 baseline 已切换到 Adapter，且 Adapter baseline 相对历史 LoRA baseline 在两套正式数据上都更优，`web demo` 的默认 `checkpoint` 与 `finetune_method` 也应同步到 Adapter 口径。

## 6. 已同步项

### 6.1 `report.md` 已同步内容

- 已将“当前正式默认仍缺少 Adapter baseline / g4 补充回归”的未来时表述改为“补充验证已完成”
- 已在默认模型与正式证据链相关位置补充轻量引用，指向 `report/adapter_verification.md`
- 已在摘要结论中明确：LoRA 的 `fixed20 / confirm_large_unseen` 结果仅保留为历史机制对照，而非当前默认模型依据

### 6.2 `web demo` 已同步内容

- 默认 `checkpoint` 已从 `workdir_multi_task/models/finetune_no_stop_lora/lora_adapters` 切换到 `workdir_multi_task/models/finetune_adapter/best_model.pth`
- 默认 `finetune_method` 已从 `lora` 切换到 `adapter`
- 默认展示组已统一为 `Adapter baseline`；`g4` 保留为可选机制验证入口，但不直接顶替 baseline 成为唯一默认
