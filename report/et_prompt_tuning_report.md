# ET Prompt 收敛实验阶段汇报

## 1. 实验背景

当前项目的默认主模型已经确定为多任务 `Adapter baseline`。在此前的多类别 prompt 改造中，我们已经完成了从“单类 YOLO tumor box + Adapter baseline”到“`WT -> TC -> ET` class-specific prompt pipeline”的切换，并得到两个明确结论：

1. 多类别 prompt 路线是有效的。  
在 `fixed20` 与 `confirm_large_unseen` 上，`class_boxes_points` 都稳定打破了历史上的三类塌缩问题，即 raw 阶段 `WT = TC = ET` 的现象已经从原先的普遍出现，下降到 `0`。

2. 当前剩余问题已经收缩到 `ET`。  
在 `confirm_large_unseen` 上，相比 `Adapter baseline`，`class_boxes_points` 的结果为：
   - `Mean Dice +0.006567`
   - `TC Dice +0.030999`
   - `WT Dice +0.002595`
   - `ET Dice -0.013895`

这说明当前多类别 prompt 链路已经不是“整体无效”，而是“WT/TC 已经成立，但 ET 还有优化空间”。因此，本轮实验不再改动 WT/TC，也不再启用 `mask_input` 分支，而是只针对 ET prompt 做小范围、低风险、可解释的收敛实验。

## 2. 实验目标

本轮目标是回答一个非常具体的问题：

> 在保持 `WT` 和 `TC` 现有 prompt 逻辑不变的前提下，是否能够仅通过调整 ET prompt 的候选区域、box、点提示与 fallback 规则，让 `ET Dice` 回升，同时不明显伤害 `WT/TC` 与整体 `Mean Dice`？

若答案为“可以”，则再推进到 `confirm_large_unseen`。  
若答案为“暂时不可以”，则说明当前阶段不应继续扩大验证范围，而应先进一步收敛 ET 机制本身。

## 3. 实验设置

### 3.1 固定不变的部分

- 主模型固定为：`Adapter baseline`
- 检测器固定为：当前单类 YOLO tumor box
- `WT` prompt 固定不变
- `TC` prompt 固定不变
- 不重训模型
- 不改后处理默认参数
- 不重新启用 `boxes + points + mask_input` 作为默认实验线

### 3.2 本轮只改 ET prompt

本轮只围绕以下几个低风险变量做 ET-only 小网格：

- `T1ce` 高亮阈值：`q90 / q92 / q95`
- ET box 放宽：`pad 0 / 2 / 4 / 8 px`
- 正点数：`1 / 2`
- 负点数：`0 / 2 / 4`
- fallback 方式：`shrunk ROI` 或 `wide ROI`

共评估 6 组配置：

1. `default`
2. `q92_pad4_p1_n2`
3. `q95_pad4_p1_n2`
4. `q95_pad8_p1_n2_widefb`
5. `q92_pad4_p2_n2`
6. `q90_pad2_p1_n0`

其中 `default` 是当前 `class_boxes_points` 的 ET 配置，用作本轮固定基线。

### 3.3 验证数据与输出位置

- 本轮先只在 `fixed20` 上做 ET 收敛
- 结果目录：`outputs/stage9_et_prompt_tuning/`
- 汇总文件：
  - `outputs/stage9_et_prompt_tuning/et_prompt_tuning.md`
  - `outputs/stage9_et_prompt_tuning/et_prompt_tuning.json`
  - `outputs/stage9_et_prompt_tuning/recommended_et_variant.md`

## 4. 结果

### 4.1 fixed20 聚合指标

| 配置 | post Mean Dice | ET post Dice | TC post Dice | WT post Dice | 相对 default 的 Mean delta | 相对 default 的 ET delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `default` | `0.547851` | `0.404318` | `0.537971` | `0.701264` | `+0.000000` | `+0.000000` |
| `q92_pad4_p1_n2` | `0.546798` | `0.399499` | `0.539554` | `0.701342` | `-0.001053` | `-0.004819` |
| `q95_pad4_p1_n2` | `0.532104` | `0.347318` | `0.543296` | `0.705699` | `-0.015747` | `-0.057000` |
| `q95_pad8_p1_n2_widefb` | `0.545067` | `0.402407` | `0.520542` | `0.712253` | `-0.002784` | `-0.001910` |
| `q92_pad4_p2_n2` | `0.547799` | `0.401877` | `0.540138` | `0.701381` | `-0.000052` | `-0.002441` |
| `q90_pad2_p1_n0` | `0.527140` | `0.341197` | `0.538925` | `0.701299` | `-0.020711` | `-0.063121` |

### 4.2 塌缩与顺序约束

本轮 6 组实验中：

- raw 阶段 `WT = TC = ET`：全部为 `0/20`
- `ET <= TC <= WT`：
  - `default / q92_pad4_p1_n2 / q95_pad4_p1_n2 / q92_pad4_p2_n2 / q90_pad2_p1_n0` 均为 `20/20`
  - `q95_pad8_p1_n2_widefb` raw 阶段仅 `3/20` 成立，说明该组虽然 post 阶段能被后处理拉回，但 raw prompt 本身已经不稳定

这说明：

- 多类别塌缩没有回来
- 但某些 ET 变体已经开始破坏 raw 阶段的类间体积顺序，属于高风险信号

### 4.3 ET fallback 使用情况

| 配置 | ET fallback 次数 | ET 小候选次数 | ET 碎片候选次数 |
| --- | ---: | ---: | ---: |
| `default` | `15` | `0` | `0` |
| `q92_pad4_p1_n2` | `15` | `0` | `0` |
| `q95_pad4_p1_n2` | `680` | `665` | `0` |
| `q95_pad8_p1_n2_widefb` | `1312` | `641` | `656` |
| `q92_pad4_p2_n2` | `15` | `0` | `0` |
| `q90_pad2_p1_n0` | `15` | `0` | `0` |

这一组统计非常关键：

- 一旦 ET 阈值升到 `q95`，fallback 次数会急剧上升
- `q95_pad4_p1_n2` 的 ET 退化，本质上不是“点提示不够”，而是“小候选太多，导致 fallback 频繁触发”
- `q95_pad8_p1_n2_widefb` 更进一步，不仅 fallback 激增，还出现了大量碎片候选，说明该组在 ET 候选生成阶段已经过于激进

## 5. 实验结论

### 5.1 当前没有找到优于 `default` 的 ET-only 版本

这是本轮最核心的结论。

虽然我们系统性地测试了 5 个 ET-only 变体，但没有任何一组能够同时满足以下条件：

1. `ET Dice` 高于当前 `default`
2. `WT` 不明显退化
3. `TC` 不明显退化
4. `Mean Dice` 不明显下降
5. 三类塌缩不回归
6. `ET <= TC <= WT` 的顺序关系不被破坏

因此，本轮结论不是“找到更优 ET 配置”，而是“当前默认 ET 配置已经是这一批低风险小修改中的最优点”。

### 5.2 当前 `default` 仍应保留为 ET 默认配置

从 fixed20 结果看：

- `default` 的 `ET post Dice = 0.404318`，仍是本轮最高
- `q92` 系列整体较稳，但 ET 都没有超过 `default`
- `q95` 系列触发了过多 fallback，导致 ET 明显下滑
- 去掉负点的 `q90_pad2_p1_n0` 直接使 ET 明显退化，说明当前 ET 任务仍然需要负点约束，不能简单删掉

所以当前阶段不宜替换掉 `default`。

### 5.3 不建议现在推进新的 ET 变体到 confirm_large_unseen

由于 fixed20 上都没有出现明确正信号，因此不建议把本轮新的 ET-only 变体推进到 `confirm_large_unseen`。

更具体地说：

- 当前不是“还没大规模验证，所以不确定”
- 而是“在 fixed20 上已经没有看到值得放大的正增益”

因此继续推进只会增加算力开销，而不会显著提升结论质量。

## 6. 对当前多类别 prompt 路线的阶段性判断

本轮实验并不否定多类别 prompt 路线，反而进一步收敛了问题边界：

1. `class_boxes_points` 路线总体成立  
它已经稳定解决了 raw 阶段三类塌缩问题，并在 `confirm_large_unseen` 上取得了 `Mean / TC / WT` 的正增益。

2. 当前瓶颈不是 WT/TC  
WT 和 TC 目前已经具备稳定的 class-specific prompt 能力，不应再大范围改动。

3. 当前核心瓶颈就是 ET  
并且 ET 的问题已经进一步定位到：
   - `T1ce` 候选区域生成
   - fallback 触发条件
   - fallback box 的构造方式

换句话说，当前问题已经不是“多类别 prompt 是否可行”，而是“ET 的最后一段细化机制还不够稳”。

## 7. 下一步建议

基于本轮结果，下一步不建议继续扩大 ET 阈值或 box padding 网格，而建议聚焦两个更有价值的方向：

### 7.1 优先改 ET fallback 机制

当前最强的异常信号来自 fallback 激增，因此下一步更应该优化：

- 什么情况下才触发 fallback
- fallback 是用 `shrunk ROI` 还是 `wide ROI`
- fallback box 是否还需要引入更稳定的 containment 约束

### 7.2 优先改 T1ce 候选生成，而不是继续加 prompt 数量

本轮结果已经说明：

- 简单增加正点数，没有带来 ET 提升
- 简单减少负点，反而会伤害 ET
- 单纯提高阈值，会把问题推向“小候选 + 高频 fallback”

因此下一步更值得做的是：

- 让 ET 候选生成本身更稳定
- 而不是继续在“多几个点、少几个点、box 再大一点”这类表面参数上来回搜索

## 8. 阶段总结

截至当前，可以给出一个清晰的阶段性结论：

> 多类别 prompt 主路线已经成立，当前主要剩余问题已经缩小为 ET；但在目前这批低风险 ET-only prompt 调整中，还没有找到优于现有 `default` 的版本，因此不建议继续推进新的 ET 变体到大规模确认集。下一步应集中火力收敛 ET fallback 与 T1ce 候选生成机制。

