# 科技实物结项说明书母版

## 项目概述

本项目围绕 BraTS 脑肿瘤数据，基于 `SAM-Med2D`、`YOLO` 与三维后处理链路，完成了从“模型训练验证”向“整病例自动分割与三维展示原型”推进。当前成果已经不再停留在单纯训练模型，而是形成了可重复运行的自动推理闭环，并积累了固定病例集的正式回归结果，可作为科技实物结项说明书的母版。

从仓库当前结果看，项目的核心价值主要体现在两点：

1. 已完成 `YOLO 检测框 -> SAM-Med2D 分割 -> 3D 后处理 -> 3D 可视化` 的整病例闭环。
2. 已在固定 `hard8` 与 `fixed20` 病例集上完成 prompt 策略、`z` 轴连续性策略与 `WT-only continuity` 的正式回归，形成了明确的默认配置建议。

## 与申报书/中期报告衔接

原目标为建设一个面向脑肿瘤三维可视化与辅助分析的 Web 系统，包含后端服务、数据库管理以及前端三维展示能力，最终形态强调“可视化展示 + 自动处理 + 交互查看”。

中期阶段已完成的内容主要包括：

- BraTS 原始数据处理与切片级数据准备；
- 基于 `YOLO + SAM-Med2D` 的自动分割主线打通；
- 初步三维重建与结果预览能力；
- 单任务与多任务微调训练流程跑通，并形成初步指标对比。

当前阶段相对中期新增并冻结的内容主要包括：

- 已将系统推进为整病例自动分割闭环，而非仅切片级实验；
- 已建立固定病例集的正式回归流程，能够对不同 prompt 策略进行可重复对比；
- 已完成 `WT-only continuity` PoC 以及 `WT gate` 收敛实验，形成默认展示配置建议。

因此，项目当前状态与申报目标的衔接关系可以概括为：算法主线和展示主线已经具备原型基础，结项阶段更适合以“展示优先、复用现有结果、适度保留在线推理”为策略，完成科技实物形式的整合与说明。

## 当前系统闭环

当前仓库已经具备如下闭环能力：

1. 数据层  
支持 BraTS 原始 `NIfTI` 病例读取、四模态输入组织、训练与推理所需的切片级数据准备。

2. 模型层  
支持 `SAM-Med2D` 的单任务与多任务微调；当前整病例自动推理使用多任务 `LoRA` 检查点。

3. 自动提示层  
通过 `YOLO` 检测器为 `SAM-Med2D` 提供自动 bbox prompt，当前正式基线工作点已固定为 `conf=0.05, iou=0.60`。

4. 整病例推理层  
已实现整病例推理入口，可对单个 BraTS 病例输出 `ET / TC / WT` 掩码及合并标签结果。

5. 三维后处理层  
已实现 3D 后处理，包含闭运算、开运算、连通域筛选、`ET ⊆ TC ⊆ WT` 约束以及 `z` 轴平滑。

6. 结果展示层  
已支持 3D HTML 预览与病例级结果回放，可直接复用现有 `outputs` 目录中的预览与汇总文件进行展示。

7. 批量验证层  
已支持对固定病例集批量生成 `summary.md`、`summary_metrics.json`、`prompt_stats.json` 等结果文件，形成正式回归链路。

综上，当前系统已经具备“样例病例自动处理 + 三维结果展示 + 结果汇总说明”的科技实物原型基础。

## 主要实验结论

### 1. 训练阶段结论

根据仓库 `README` 中已记录的训练日志，当前可确认的训练阶段结果如下：

| 配置 | 最佳验证 Dice | 最佳验证 IoU | 结论 |
| --- | ---: | ---: | --- |
| 单任务 Adapter | 0.8335 | 0.7402 | 当前训练日志中表现最好，收敛也最快 |
| 单任务 LoRA | 0.8113 | 0.6966 | 明显优于原始基线 |
| 多任务 LoRA | 0.7265 | 0.6215 | 低于单任务，但符合多类别任务更复杂的预期 |
| 原始基线 | 0.5303 | 0.3882 | 作为对照参考 |

训练阶段结论可以冻结为：微调本身是有效的，模型能力上限已被验证；当前系统瓶颈已经从“模型是否能学到”转移到“自动 prompt 质量与整病例推理策略是否稳定”。

### 2. 系统阶段结论

当前系统已从单纯训练模型推进到 `YOLO 检测框 -> SAM-Med2D 分割 -> 3D 后处理 -> 3D 可视化` 的整病例闭环。这一结论已经由仓库现有整病例推理、后处理、HTML 预览与批量回归结果共同支撑。

### 3. 正式 baseline 结论

当前正式 baseline 已固定为：

- detector: `conf=0.05, iou=0.60`
- prompt policy: `top1`
- 后处理：`closing_radius=2, opening_radius=1, wt_keep_largest=true, keep_topk_tc=1, keep_topk_et=1, z_smooth_iterations=3`

在固定 20 例正式回归上，baseline 结果为：

- `post Mean Dice = 0.524049`
- `ET post Dice = 0.372901`
- `TC post Dice = 0.515366`
- `WT post Dice = 0.683878`

这组配置应视为当前全项目的正式对照基线。

### 4. 全类 smooth / interpolate 结论

围绕全类别 `z` 轴连续性 prompt 所做的 `smooth` 与 `interpolate` 实验，已经可以冻结为“无稳定收益”。

固定 20 例结果如下：

| 配置 | Post Mean Dice | ET | TC | WT | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline top1 | 0.524049 | 0.372901 | 0.515366 | 0.683878 | 正式基线 |
| `top1 + smooth` | 0.5233 | 0.3794 | 0.5206 | 0.6700 | 整体略降，WT 回退 |
| `top1 + interpolate` | 0.5241 | 0.3725 | 0.5150 | 0.6849 | 基本持平，收益极弱 |

结合 `hard8` 与 `fixed20` 的正式回归，结论为：

- 全类 `smooth` 不适合作为默认策略；
- 全类 `interpolate` 虽较安全，但没有形成足够明确的整体收益；
- 不建议再回到“全类 continuity”方向继续大规模扫参。

### 5. Prompt policy 结论

在 `top1`、`top2_merge`、`ET/TC=top1, WT=top2_merge` 等 prompt 策略对比后，当前结论仍然保持不变：默认 prompt policy 应保持 `top1`。已有结果显示，`top2_merge` 或混合策略在困难病例上可能出现局部增益，但在固定 20 例正式回归中不能稳定泛化，因此不适合作为默认方案。

### 6. WT-only continuity 结论

在保持 `ET/TC` 完全不变、仅对 `WT` 启用 continuity 的前提下，`WT-only continuity` 已经验证成立。

冻结结果如下：

- `hard8`：baseline `0.365315 -> g0 0.380571`
- `hard8 WT`：`0.636206 -> 0.681975`
- `fixed20`：baseline `0.524049 -> g0 0.530754`
- `fixed20 WT`：`0.683878 -> 0.703994`

对应解释为：

- `ET/TC` 保持 baseline 行为不变；
- 整体收益主要来自 `WT`；
- 该收益不是由 box smooth 带来，而是由 `WT missing_box` 场景下的补救机制带来。

因此，`WT-only continuity` 是当前项目在 prompt 机制层面最明确成立的新增结果。

### 7. WT gate 收敛结论

在 `WT-only continuity` 的基础上，进一步做了 `WT gate` 参数收敛，目标是降低误触发与 `harm`，同时保持 overall 与 WT 的正收益。

现有结果中，`g4` 更适合作为默认展示配置：

- `fixed20 post Mean Dice = 0.528580`，较 baseline `+0.004531`
- `fixed20 WT post Dice = 0.697473`，较 baseline `+0.013594`

与高收益参考组 `g0` 相比，`g4` 的优势不在于绝对数值更高，而在于误触发显著减少，整体更稳：

- `trigger_total: 1314 -> 1098`
- `harm: 733 -> 671`
- `low_score: 385 -> 176`
- `center_jump: 133 -> 72`
- `area_jump: 205 -> 138`

同时，`missing_box` 主收益仍被保留：

- `missing_box: 827 -> 811`

因此，当前可以冻结的机制结论是：

1. 主要收益来自 `WT missing_box` 补救；
2. 应优先收紧 `low_score / center_jump / area_jump` 误触发；
3. 不建议再回到 `ET/TC continuity` 或大规模全局扫参。

### 8. confirm_large_unseen 大样本确认结论

在固定 20 例之外，项目又补做了 `confirm_large_unseen` 大样本确认实验。该集合包含 167 例病例，来源于当前验证集全部 187 例中排除 `fixed20` 与 `hard8` 后得到的更大未参与调参样本。

本轮只比较 baseline 与 `g4`，不再调参、不改模型、不改代码逻辑。结果如下：

- baseline：`post Mean Dice = 0.536181`，`ET = 0.432519`，`TC = 0.555623`，`WT = 0.620401`
- `g4`：`post Mean Dice = 0.535515`，`ET = 0.432519`，`TC = 0.555623`，`WT = 0.618403`
- 相对 baseline，`g4` 的 overall 变化为 `-0.000666`，`WT` 变化为 `-0.001998`
- case-level 统计为 `win / tie / loss = 86 / 13 / 68`
- mean delta 为 `-0.000666`，bootstrap 95% CI 为 `[-0.005824, 0.004504]`

这一结果说明：`g4` 在 fixed20 上的收益结论并没有在更大未见样本上稳定复现。虽然 `ET/TC` 仍保持不退化，但 `WT` 与 overall 均出现轻微回退，因此当前不能再将 `g4` 表述为“更大样本默认最优配置”。

结合 fixed20 与 `confirm_large_unseen` 两轮证据，可以冻结的判断是：

1. `WT-only continuity` 作为机制方向是成立的；
2. `g4` 可作为 `WT missing_box` 补救机制的展示配置；
3. 若以更大未参与调参样本作为结项主证据，baseline 仍是更稳妥的默认主对照。

## 推荐默认配置

结合 fixed20 与 `confirm_large_unseen` 的现有结果，当前更适合冻结三套角色清晰的配置：baseline 作为正式主对照与稳妥默认，`g4` 作为 `WT-only continuity` 展示参考组，`g0` 作为高收益研究参考组。

### 1. 正式主对照与稳妥默认配置

- detector: `conf=0.05, iou=0.60`
- prompt policy: `top1`
- `z_prompt_mode: none`
- `WT continuity: disabled`
- 后处理：`closing_radius=2, opening_radius=1, wt_keep_largest=true, keep_topk_tc=1, keep_topk_et=1, z_smooth_iterations=3`

这套配置对应当前正式 baseline。原因很明确：在更大未参与调参样本 `confirm_large_unseen` 上，baseline 的 overall 与 WT 均优于 `g4`，因此更适合作为结项阶段的主对照与稳妥默认。

### 2. WT-only continuity 展示参考配置

- detector: `conf=0.05, iou=0.60`
- prompt policy: `top1`
- `z_prompt_mode: none`
- `WT continuity: enabled`
- `WT gate score_thresh = 0.08`
- `WT gate center_shift_max = 72`
- `WT gate area_ratio_min = 0.33`
- `WT gate area_ratio_max = 3.0`
- `WT gate mask_dilate_iters = 1`
- `WT gate mask_blur_kernel = 3`
- 后处理：`closing_radius=2, opening_radius=1, wt_keep_largest=true, keep_topk_tc=1, keep_topk_et=1, z_smooth_iterations=3`

这套配置对应 `g4`。它在 fixed20 上表现为更稳健的 `WT gate` 收敛组，适合用于展示 `WT-only continuity` 机制本身；但在 `confirm_large_unseen` 上未能继续优于 baseline，因此更适合作为“机制展示参考组”，而不是结项阶段唯一默认配置。

### 3. 高收益参考组

保留 `g0` 作为高收益参考组：

- `WT gate score_thresh = 0.15`
- `WT gate center_shift_max = 48`
- `WT gate area_ratio_min = 0.5`
- `WT gate area_ratio_max = 2.0`
- `WT gate mask_dilate_iters = 1`
- `WT gate mask_blur_kernel = 3`

`g0` 的 fixed20 结果更高，为：

- `post Mean Dice = 0.530754`
- `WT post Dice = 0.703994`

但其误触发与 `harm` 明显更多，因此更适合作为研究参考组，而不是默认展示组。

## 结项展示版 Web 系统方案

结项展示版 Web 系统建议采用“展示优先、复用现有结果、在线推理可选”的策略，不建议在结项阶段重写算法主线。

### 1. 展示优先目标

优先实现以下能力：

1. 样例病例回放  
可直接选择固定样例病例，查看原始结果、后处理结果、三维预览与关键指标。

2. 结果查看  
可查看 `summary.md`、`summary_metrics.json`、病例级指标、关键配置与 HTML 三维预览。

3. 实验对比  
可对比 baseline、`smooth`、`interpolate`、`WT-only continuity g0`、`WT gate g4` 等代表性方案。

### 2. 在线推理定位

在线推理建议作为可选功能，而非结项展示主线。原因是：

- 现有项目价值已经足以通过固定样例与正式回归结果展示；
- 在线推理需要额外考虑 GPU 环境、任务排队、文件上传与耗时反馈；
- 结项阶段更应优先保证展示稳定性与材料完整性。

### 3. 实现原则

展示版 Web 系统应尽量复用仓库现有产物，不重写算法主线：

- 直接复用 `outputs` 目录中的 `summary.md`、`summary_metrics.json`、`prompt_stats.json`；
- 直接复用现有 `preview_3d_compare_all.html`；
- 后端仅做文件索引、结果查询与静态页面组织；
- 数据库可只存病例元信息、结果索引和说明文本，避免在结项阶段引入不必要复杂度。

### 4. 建议展示内容

建议选取以下内容作为展示主线：

- baseline 与 `WT gate g4` 的 fixed20 汇总对比；
- 2 到 3 个典型病例的三维预览；
- 1 个 hard case 的 `WT missing_box` 补救前后示意；
- 系统闭环流程图与默认配置说明。

## 当前不足与下一步

当前项目的主要不足有以下几点：

1. Web 端完整产品形态尚未最终封装  
原目标中的后端、数据库、前端三维展示虽已有清晰路径，但当前仓库中更成熟的是算法闭环和离线结果展示，而不是完整在线系统。

2. WT continuity 仍存在误触发  
虽然 `g4` 已较 `g0` 明显收敛，但 `harm` 仍然存在，说明 `missing_box` 补救虽有效，但长段传播和边界条件仍需进一步控制。

3. 机制收益集中在 WT  
当前实验表明，`ET/TC` continuity 不适合继续推进，收益主要集中在 `WT`。这意味着项目现阶段的算法优化空间更偏向细化 `WT gate`，而不是再做更大范围 prompt 扩展。

4. 结果说明材料仍需整理成结项交付件  
仓库中已有大量实验结果，但还需要进一步压缩为面向导师评审与结项提交的说明材料、截图和展示页面。

下一步建议如下：

1. 以 baseline 作为结项主对照配置，按需展示 `g4` 作为 `WT-only continuity` 的机制样例。
2. 保留 `g0` 作为研究参考组，用于说明“高收益但误触发更多”的技术权衡。
3. 若仍有少量实验空间，优先分析 `WT missing_box` 长段传播导致的 `harm` 病例，不建议再回到全类 continuity 或大规模扫参。
4. 将当前结果整理为图文并茂的结项材料，而不是继续扩展新的算法分支。

## 结项材料清单

建议按“科技实物”路线准备以下结项材料：

1. 软件原型  
包含结项展示版 Web 系统或可运行的本地展示原型，能够完成病例浏览、结果查看与三维预览。

2. 使用说明  
提供不少于 1000 字的中文使用说明，内容包括环境要求、目录结构、启动方法、病例查看方式、结果解释与注意事项。

3. 运行截图或压缩包  
至少准备 2 到 3 张运行截图，或提供包含 HTML 预览与结果页面的压缩包，便于答辩与提交。

4. 核心结果说明  
整理一份简明结果说明，冻结以下内容：
- 系统闭环已经完成；
- baseline 已固定；
- 全类 `smooth / interpolate` 无稳定收益；
- `WT-only continuity` 已成立；
- `WT gate g4` 适合作为默认展示配置；
- 当前主要收益来自 `WT missing_box` 补救。

5. 待补材料

- 若结项要求提供完整 Web 部署文档、数据库表设计或演示视频，当前仓库内对应材料待补；
- 若需补充更多训练曲线图、病例级对比图或正式答辩 PPT，当前可基于现有 `outputs` 与 `README` 继续整理。
