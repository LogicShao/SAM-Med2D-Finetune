# 科技实物结项说明书（阶段同步版）

> 同步时间：2026-03-23
> 本文档用于同步仓库当前最新实验进展、默认展示口径与结项材料状态。

## 1. 项目概述

本项目面向 BraTS 脑肿瘤分割任务，基于 `YOLO + SAM-Med2D + 3D 后处理` 构建了从训练、自动提示、整病例推理到三维展示的完整闭环。到当前阶段，项目已经不再停留在单纯的训练曲线比较，而是形成了可重复运行、可批量验证、可用于结项展示的整病例自动分割原型。

当前可以明确冻结的项目状态如下：

- 训练阶段已经完成单任务与多任务微调对比，默认主模型应切换为多任务 `Adapter`。
- 正式回归链路已经覆盖 `fixed20`、`confirm_large_unseen` 等数据集，能够稳定输出 `ET / TC / WT` 三类结果与病例级汇总指标。
- `WT-only continuity` 的机制验证已经补齐到 `Adapter` 口径，但当前仍应保留 `baseline` 作为默认标准入口，`g4` 作为机制增强对照组。
- 多类别 Prompt 改造已经完成从“统一 YOLO 框”到“类间差异化 Prompt” 的切换，类间塌缩问题已经解决。
- ET Prompt 专项调优已经完成一轮系统性收敛实验，当前没有发现优于 `default` 的 ET-only 新变体。
- 结项图表、定性示意图与差异化 Prompt 流程图已经整理完成，可直接用于报告或答辩展示。

## 2. 当前系统闭环

仓库当前已经具备如下能力：

1. 数据层：支持 BraTS 原始 `NIfTI` 病例读取、四模态组织与切片级训练/推理数据准备。
2. 模型层：支持 `SAM-Med2D` 的单任务与多任务微调，当前正式默认模型为多任务 `Adapter` 检查点。
3. 自动提示层：通过 `YOLO` 检测器为 `SAM-Med2D` 自动生成初始框提示，并进一步支持类间差异化 Prompt。
4. 整病例推理层：支持对单个 BraTS 病例输出 `ET / TC / WT` 掩码、融合标签与病例级评估结果。
5. 3D 后处理层：支持闭运算、开运算、连通域筛选、`ET ⊆ TC ⊆ WT` 约束与 `z` 向平滑。
6. 结果展示层：支持病例级 3D HTML 预览、总结指标、提示统计与可复用的结项图表。
7. 批量验证层：支持固定病例集的正式回归，自动生成 `summary.md`、`summary_metrics.json`、`prompt_stats.json` 等文件。

## 3. 训练阶段结论

根据当前仓库已冻结的训练日志，训练阶段可以保留如下结论：

| 配置 | 最佳验证 Dice | 最佳验证 IoU | 结论 |
| --- | ---: | ---: | --- |
| 单任务 WT Adapter | 0.8335 | 0.7402 | 单任务 `WT` 二值分割的最佳历史结果 |
| 单任务 WT LoRA | 0.8113 | 0.6966 | 明显优于原始基线 |
| 多任务 Adapter | 0.7560 | 0.6619 | 当前多任务训练口径下的最佳主模型 |
| 多任务 LoRA | 0.7265 | 0.6215 | 低于多任务 Adapter |
| 原始基线 | 0.5303 | 0.3882 | 对照项 |

训练阶段的核心结论没有变化：模型本身已经学到有效分割能力，当前系统瓶颈主要不在“模型能否学到”，而在“自动 Prompt 质量、整病例推理与后处理链路是否足够稳定”。

## 4. 最新实验进展

### 4.1 Stage7：Adapter 正式补齐验证

本轮最重要的同步点，是正式补齐了 `Adapter` 在 `fixed20` 与 `confirm_large_unseen` 上的 baseline / g4 证据链。历史 LoRA 结果现在只保留为对照，不再作为默认模型依据。

| 数据集 | 模型 | 组别 | post Mean Dice | ET | TC | WT |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `fixed20` | Adapter | baseline | 0.528955 | 0.374207 | 0.512041 | 0.700616 |
| `fixed20` | Adapter | g4 | 0.536124 | 0.374207 | 0.512041 | 0.722123 |
| `fixed20` | LoRA | baseline | 0.524049 | 0.372901 | 0.515366 | 0.683878 |
| `fixed20` | LoRA | g4 | 0.528580 | 0.372901 | 0.515366 | 0.697473 |
| `confirm_large_unseen` | Adapter | baseline | 0.546043 | 0.434018 | 0.556753 | 0.647360 |
| `confirm_large_unseen` | Adapter | g4 | 0.547477 | 0.434018 | 0.556753 | 0.651660 |
| `confirm_large_unseen` | LoRA | baseline | 0.536181 | 0.432519 | 0.555623 | 0.620401 |
| `confirm_large_unseen` | LoRA | g4 | 0.535515 | 0.432519 | 0.555623 | 0.618403 |

Adapter 口径下可以冻结的结论如下：

- `Adapter baseline` 在 `fixed20` 与 `confirm_large_unseen` 上均优于历史 `LoRA baseline`。
- `Adapter g4` 在两个数据集上也都取得正增益，其中 `fixed20` 的 `WT` 增益更明显。
- 但从角色定位上，`baseline` 仍应保留为默认标准入口，`g4` 更适合作为 `WT-only continuity` 的机制展示组，而不是唯一默认方案。
- `confirm_large_unseen` 上 `Adapter g4` 相对 `Adapter baseline` 的 case-level 统计为 `86 / 7 / 74 (win / tie / loss)`，mean delta 为 `+0.001433`，说明其增益真实存在，但幅度较小，更适合在报告中解释为“可解释的小幅增强”，而非“完全替代 baseline 的新默认方案”。

详细补充证据见 [adapter_verification.md](./adapter_verification.md)。

### 4.2 Stage8：多类别 Prompt 消融结论

类间差异化 Prompt 的主要任务，不是简单追求某一类指标上升，而是先解决三类输出塌缩问题，并在此基础上尽量提升整体指标。`fixed20` 上的正式对比如下：

| 方案 | fixed20 post Mean Dice | ET | TC | WT | raw all_equal_ratio | post all_equal_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Adapter baseline` | 0.528955 | 0.374207 | 0.512041 | 0.700616 | 1.0 | 1.0 |
| `class_boxes` | 0.492784 | 0.248299 | 0.528754 | 0.701298 | 0.0 | 0.0 |
| `class_boxes_points` | 0.547851 | 0.404318 | 0.537971 | 0.701264 | 0.0 | 0.0 |
| `class_boxes_points_mask` | 0.526766 | 0.349253 | 0.530001 | 0.701045 | 0.0 | 0.0 |

对 `fixed20` 而言，当前最优结论很明确：

- `baseline` 虽然整体还能工作，但 `WT = TC = ET` 的类间塌缩在 raw/post 两个阶段都达到 `100%`，不适合作为“多类别分析模式”。
- `class_boxes` 解决了塌缩，但 ET 退化明显，单独使用不划算。
- `class_boxes_points` 是当前最佳平衡点，相对 `Adapter baseline` 的增益为：`Mean Dice +0.018896`、`ET +0.030110`、`TC +0.025930`、`WT +0.000649`。
- `class_boxes_points_mask` 没有进一步带来稳定收益，说明在当前阶段继续堆叠 mask prompt 的收益有限。

在 `confirm_large_unseen` 上，当前正式跑通的是 `Adapter + class_boxes_points`，结果为：

- `post Mean Dice = 0.552610`
- `ET = 0.420123`
- `TC = 0.587752`
- `WT = 0.649955`

相对 `Adapter baseline` 的变化为：

- `Mean Dice +0.006567`
- `ET -0.013895`
- `TC +0.030999`
- `WT +0.002595`

因此可以冻结的结论是：

- 多类别差异化 Prompt 路线已经成立，三类塌缩问题已经解决。
- 当前 `WT` 与 `TC` 的类间 Prompt 设计是有效的，且在更大未见样本上仍有正增益。
- 剩余主要问题已经收敛到 `ET`，后续优化应聚焦 ET，而不是重新推翻整条多类别 Prompt 路线。

### 4.3 Stage9：ET Prompt 收敛实验结论

在 Stage8 之后，本轮只对 ET Prompt 做小范围、低风险、可解释的收敛实验，目标是在不动 WT/TC 逻辑的前提下，寻找优于当前 `default` 的 ET-only 版本。

`fixed20` 上的正式结果如下：

| ET 变体 | post Mean Dice | ET | TC | WT |
| --- | ---: | ---: | ---: | ---: |
| `default` | 0.547851 | 0.404318 | 0.537971 | 0.701264 |
| `q92_pad4_p1_n2` | 0.546798 | 0.399499 | 0.539554 | 0.701342 |
| `q95_pad4_p1_n2` | 0.532104 | 0.347318 | 0.543296 | 0.705699 |
| `q95_pad8_p1_n2_widefb` | 0.545067 | 0.402407 | 0.520542 | 0.712253 |
| `q92_pad4_p2_n2` | 0.547799 | 0.401877 | 0.540138 | 0.701381 |
| `q90_pad2_p1_n0` | 0.527140 | 0.341197 | 0.538925 | 0.701299 |

本轮 ET 收敛实验的核心结论是：

- 当前没有任何一个 ET-only 变体同时优于 `default` 的 `Mean Dice` 和 `ET Dice`。
- `default` 仍是当前最稳妥的 ET 默认配置。
- `q95` 系列会显著拉高 fallback 频率，其中 `q95_pad4_p1_n2` 的 ET fallback 次数达到 `680`，`q95_pad8_p1_n2_widefb` 更达到 `1312`，说明阈值过高会把问题推向“小候选 + 高频 fallback”。
- `q90_pad2_p1_n0` 去掉负点后 ET 明显退化，说明当前 ET 分支仍需要负点约束。

因此可以冻结的结论是：

- `ET default` 保持不变。
- 当前不建议把新的 ET-only 变体推进到 `confirm_large_unseen`。
- 后续如果继续优化，应优先改 ET 候选区域生成与 fallback 机制，而不是继续做大范围阈值/点数网格搜索。

详细说明见 [et_prompt_tuning_report.md](./et_prompt_tuning_report.md)。

## 5. 当前默认方案与展示口径

结合 Stage7、Stage8、Stage9 的最新证据，当前建议冻结如下展示口径：

### 5.1 标准模式默认方案

- 模型：多任务 `Adapter baseline`
- Prompt：`yolo_box + top1`
- `z_prompt_mode = none`
- `WT continuity = disabled`
- 后处理：`closing_radius=2, opening_radius=1, wt_keep_largest=true, keep_topk_tc=1, keep_topk_et=1, z_smooth_iterations=3`

这个方案用于：

- 正式默认回归口径
- web demo 默认入口
- 结项说明中的“标准模式”

### 5.2 多类别分析模式

- 模型：多任务 `Adapter`
- Prompt 变体：`class_boxes_points`
- ET 变体：`default`

这个方案用于：

- 展示差异化 Prompt 策略的有效性
- 演示 `WT / TC / ET` 三类在定性图中的层级分布
- 支撑 Prompt 机制相关的图表与说明

### 5.3 机制验证对照组

- 方案：`Adapter g4`
- 定位：`WT-only continuity` 的机制增强组
- 用途：展示 `WT missing_box` 补救思路与小幅稳定增益

需要强调的是：

- `LoRA baseline / g4` 现在只保留为历史对照，不再作为当前默认模型依据。
- `Adapter baseline` 是正式默认标准。
- `Adapter g4` 是机制展示组，不应在文档中表述成“唯一最优默认方案”。

## 6. 结项图表与展示素材状态

当前已经完成并落盘的结项图表如下：

- [图1：主路线与结果总览](../fig/png/figure1_main_model_route.png)
- [图2：多类别 Prompt 修复对比](../fig/png/figure2_multiclass_prompt_repair.png)
- [图3：分类别增量对比](../fig/png/figure3_classwise_delta.png)
- [图4：ET Prompt 调优结果](../fig/png/figure4_et_prompt_tuning.png)
- [图5：Adapter 与 LoRA 训练对比](../fig/png/figure5_training_adapter_vs_lora.png)
- [图6：YOLO 指标摘要](../fig/png/figure6_yolo_summary.png)
- [图7：YOLO 提示框 + SAM 三类预测定性示意](../fig/png/figure7_yolo_sam_qualitative_demo.png)
- [图8：差异化 Prompt 策略流程图](../fig/png/figure8_prompt_strategy_flow.png)

对应的 SVG 版本也已整理至 `fig/svg/` 目录，可直接用于论文、答辩或高分辨率排版。

## 7. Web demo 同步状态

当前结项展示更适合采用“展示优先、复用现有结果、默认口径前置”的策略，而不是在结项阶段继续扩展新的在线推理能力。当前建议如下：

- web demo 默认模型切换到 `Adapter baseline`。
- 标准入口绑定“标准模式”，多类别分析作为辅助入口保留。
- g4 作为可选对照入口存在，但不应覆盖 baseline 的默认地位。
- 页面展示优先复用现有 `outputs/` 结果、3D HTML 预览与图表素材，保证稳定性与可讲解性。

## 8. 当前不足与下一步建议

当前仍然存在的主要问题如下：

1. `WT-only continuity` 仍存在 case-level `harm`，虽然在 Adapter 口径下整体可保留，但还不适合直接升级为唯一默认方案。
2. 多类别 Prompt 路线已经成立，但 ET 仍是主要瓶颈，特别是候选区域生成与 fallback 触发逻辑。
3. 当前最值得继续投入的方向，不是重新回到全类别 continuity 或大规模扫参，而是集中分析：
   - `WT missing_box` 造成的误触发病例；
   - ET 候选区域质量；
   - ET fallback 的触发条件与 fallback box 生成方式。

因此，下一步建议冻结为：

- 默认展示与正式报告全部切换到 `Adapter` 口径。
- 以 `Adapter baseline` 作为主对照，以 `Adapter g4` 作为机制增强对照。
- 多类别 Prompt 维持 `class_boxes_points + ET default`。
- 若仍有少量实验空间，优先分析 ET fallback 与 WT missing_box，而不是继续扩展新的大规模参数网格。

## 9. 关键文件索引

- 总报告：[report.md](./report.md)
- Adapter 补充验证：[adapter_verification.md](./adapter_verification.md)
- ET Prompt 收敛报告：[et_prompt_tuning_report.md](./et_prompt_tuning_report.md)
- confirm_large_unseen 历史确认报告：[confirm_large_unseen_confirmation.md](./confirm_large_unseen_confirmation.md)
- Adapter 对比汇总 JSON：`outputs/stage7_adapter_verification/summary/adapter_comparison.json`
- ET Prompt 汇总 JSON：`outputs/stage9_et_prompt_tuning/et_prompt_tuning.json`

## 10. 阶段结论

截至当前仓库状态，可以给出一条清晰、统一的结项结论：

> 项目已经完成从训练验证到整病例自动分割闭环的推进，并在 `Adapter` 口径下补齐了正式 baseline/g4 证据链。多类别差异化 Prompt 路线已经成立，当前剩余主要问题已收敛到 ET 分支；因此，结项阶段应以 `Adapter baseline` 作为默认标准方案，以 `class_boxes_points + ET default` 作为多类别分析方案，以 `Adapter g4` 作为 `WT-only continuity` 的机制展示方案，围绕现有图表与定性结果完成最终展示与说明。
