# 建立实验报告记录项目级技能

## Goal

为训练、评估和消融实验完成后的结果记录建立一个简洁、可追溯且面向 ICASSP 2027 主线的项目级 Codex skill，使 agent 能稳定产出可用于实验决策与论文材料沉淀的报告。

## What I already know

* 项目当前主线为“ICASSP 2027: 层级提示鲁棒脑肿瘤分割研究”。
* 现有仓库已包含训练、全卷推理、批量验证命令以及 `report/` 目录中的实验报告痕迹。
* 用户希望报告简洁、突出重点、与主线匹配，并尽可能抽象为项目级 skill。
* 主线 PRD 将原始患者级 ET/TC/WT 宏 Dice 定义为不可替代的主决策终点；后处理指标只能作为次级受控分析。
* 当前批量评估器已稳定写出 `summary_metrics.json`、`summary_metrics.csv` 和逐病例指标，适合作为报告的结构化证据源。
* `.agents/skills/` 是本项目跨平台共享的本地 skill 层；现有 `autodl-rsync-sync` 使用简洁的 `SKILL.md` 加可选脚本资源的结构。

## Assumptions

* 项目级 skill 不修改训练或验证主流程，而是作为 agent 的实验完成门槛自动触发。
* 报告应引用已有的命令、配置、日志、指标和产物路径，避免复制大体量原始日志。

## Requirements

* 定义实验完成后的最小报告结构与必填证据。
* 使报告内容能直接支撑主线的基线比较、消融和论文统计。
* 将工作流封装为可复用的项目级 skill，并避免与现有 Trellis 收尾流程重复。
* 首版同时生成单次事实报告和极简跨实验决策索引。
* 成功、失败、中断和证据不完整的实验都必须生成报告并进入决策索引。
* 非成功实验至少记录失败阶段、失败原因、最后有效证据、已产生的产物、重跑建议和明确状态。
* 训练、评估、消融或诊断实验进入终态后，agent 默认执行该 skill；只有用户明确要求跳过时才不记录。
* skill 必须优先读取运行 `manifest.json` 和结构化指标，其次读取配置与日志；缺失信息写为 `missing`，不得推断。
* skill 不得自动修改主线 PRD、冻结协议或论文主张，只能生成证据报告与决策索引。
* 报告与决策索引必须先写入本地 Git 工作区；AutoDL 只接收经 SHA-256
  校验的 rsync 副本，禁止 agent 通过 SSH 或远端脚本直接编辑报告。

## Repository Findings

* 既有报告已覆盖实验目的、固定设置、实际命令、聚合指标、结果路径和结论，但不同报告的主指标口径不一致，且有将后处理 Dice 用作主结论的历史写法。
* 主线要求正式实验额外记录原始 ET/TC/WT Dice、HD95、配对不确定性、提示鲁棒性、最差四分位、零 Dice/严重失败、层级违规和 raw/post 差异；具体字段应按实验类型分级，不能要求每次小排错实验填写完整论文表。
* 每次正式运行还应能追溯命令/解析后配置、数据与病例集、随机种子、模型与检查点哈希、评估器版本、产物路径和是否满足候选晋级或回退门槛。

## Feasible Approaches

### A. 仅维护 Markdown 模板

* Agent 复制固定模板并手工读取结果。
* 优点：零脚本、最低初始成本。
* 局限：字段容易遗漏，难以约束 raw/post 口径和证据路径。

### B. 项目级 `record-experiment-report` skill（推荐）

* 每次训练或评估进入终态后默认调用 skill；skill 读取结构化指标与已有产物，按实验类型生成一份可追溯的 Markdown 报告，并输出明确的“晋级 / 保持 / 排除 / 重跑 / 证据不足”结论。
* skill 不改动训练或评估入口，不引入服务或数据库；缺失证据必须标注为 `missing`，不得推测或把 post 指标写成主终点。
* 单次报告写入实验所属 Trellis 任务的 `research/experiments/<run_id>.md`；跨实验索引写入最上层研究任务的 `research/experiment-index.md`。

### C. 结构化运行清单 + 自动汇总器

* 在 B 的基础上再维护 JSONL/CSV 运行注册表，并由脚本自动生成对比总表和论文候选表。
* 优点：多 seed、多阶段、论文表格的汇总成本最低。
* 局限：需要先冻结 run 元数据与评估输出 schema；首版实现和维护成本更高。

## Recommended MVP

采用 B，并仅增加一个极简的跨实验“决策索引”而非完整指标数据库。每份单次报告是 Git 内的研究结论入口，运行目录中的 manifest、指标和日志仍是底层证据源。索引只保留 `run_id`、实验类型、对照组、原始宏 Dice delta、关键守卫指标、运行状态、决策和报告链接。

## Confirmed Decisions

* 首版范围选择“项目级 `record-experiment-report` skill + 单次事实报告 + 极简决策索引”。
* 暂不实现完整 JSONL/CSV 实验数据库，也不自动生成论文总表。
* 失败、中断或证据不完整的实验同样必须留档并进入索引，以避免重复消耗算力并保留路线排除依据。
* 实验报告是 agent 完成实验的默认门槛，除非用户明确要求跳过。
* 报告归属实验子任务，索引归属最上层研究任务；`report/` 保留给最终对外报告和论文材料。
* 用户已于 2026-07-14 确认完整需求与技术方案。
* 用户于 2026-07-14 补充确认本地优先规则：先在本地生成报告和索引，
  再逐文件 rsync 到 AutoDL；同步冲突不得通过远端编辑绕过。

## Technical Approach

### Skill Structure

```text
.agents/skills/record-experiment-report/
  SKILL.md
  agents/openai.yaml
  references/report-contract.md
  assets/experiment-report-template.md
```

首版不增加自动汇总脚本。当前运行 manifest 仍在建设中，先由 skill 使用结构化文件和模板完成报告；待 schema 冻结且出现重复手工逻辑后，再按 DRY 原则增加确定性脚本。

### Evidence Precedence

1. 运行目录中的 `manifest.json`、哈希和退出状态。
2. `summary_metrics.json`、`summary_metrics.csv`、逐病例指标和统计产物。
3. 解析后配置、命令、训练曲线和日志。
4. 人工说明仅作补充，不得覆盖结构化证据。

### Report Layout

1. `Verdict`：运行状态、决策和一句话结论。
2. `Reproducibility`：run ID、目的、代码状态、数据、seed、命令/配置、checkpoint 与证据路径。
3. `Primary Results`：raw 患者级宏 Dice、ET/TC/WT、对照 delta；正式实验追加置信区间或配对统计。
4. `Guardrails`：按实验类型追加 HD95、提示鲁棒性、失败率、层级违规、效率或 raw/post 差异。
5. `Failures and Missing Evidence`：失败阶段、缺失字段和结论限制。
6. `Decision and Next Action`：晋级、保持、排除、重跑或证据不足，以及唯一明确的下一步。
7. `Artifact Links`：仅列路径和哈希，不复制日志或大产物。

报告以表格和短结论为主，不复述运行日志。诊断实验只填写相关守卫指标；正式候选、基线和 locked-test 才启用完整论文证据字段。

### Task Ownership

* 单次报告：`.trellis/tasks/<owning-task>/research/experiments/<run_id>.md`。
* 决策索引：沿 `task.json.parent` 找到最上层研究任务，写入其 `research/experiment-index.md`；没有父任务时写入当前任务。
* 运行证据：继续保存在 `<run-root>/<run-id>/`，不复制到 Git。
* 同一 run ID 重复执行 skill 时应保持幂等；若恢复运行产生新证据，只更新状态并保留简短变更记录，不静默覆盖旧结论。

## Decision (ADR-lite)

**Context**：项目已有结构化评估产物和 Trellis 研究任务，但实验结论分散在运行目录与历史报告中，且存在 raw/post 口径漂移风险。

**Decision**：新增共享项目 skill，将实验留档设为默认完成门槛，采用“任务内单次事实报告 + 根研究任务极简决策索引”，运行目录继续保存机器证据。

**Consequences**：首版保持简洁且不侵入训练代码；报告可直接服务主线决策。代价是 agent 仍需按证据契约读取多种文件，待 manifest schema 稳定后再评估脚本化。

## Implementation Plan

1. 创建并校验 `.agents/skills/record-experiment-report/` 基础结构与触发描述。
2. 编写精简 `SKILL.md`、详细报告契约和 Markdown 模板。
3. 使用一个现有成功实验和一个缺失证据场景进行前向验证，修正触发、字段与幂等规则。
4. 运行 Trellis 质量检查；确认无需修改主线 PRD、训练或评估代码。

## Expansion Sweep

* 未来演进：当多 seed 和 locked-test 开始后，可在不迁移旧报告的前提下，将决策索引升级为由结构化 manifest 自动生成。
* 关联场景：训练完成、批量验证完成、AutoDL 拉取结果后三类入口都应调用同一 skill；Trellis 的 `finish-work` 只做代码任务收尾，不应代替实验结论记录。
* 失败边界：结果目录、基线、哈希、病例集或主终点缺失时，报告必须保留失败状态与证据路径，禁止生成“成功”结论；同名 run 不覆盖旧报告。

## Acceptance Criteria

* [x] skill 能从一个成功运行产物生成单次报告，并在根研究任务索引中新增唯一条目。
* [x] skill 能为失败、中断或证据不完整的运行生成精简报告，不产生成功结论。
* [x] 主结果始终以 raw 患者级宏 Dice 为首要口径，post 指标只能位于次级分析。
* [x] 报告包含复现字段、证据路径、结论限制、决策和下一步，且不复制大体量日志。
* [x] 同一 run ID 重复记录不会产生重复索引行；恢复运行不会静默删除旧结论。
* [x] skill 不修改训练、推理、评估代码、主线 PRD 或冻结协议。
* [x] skill 目录通过 skill 基础校验，并使用一个现有实验产物完成前向验证。
* [x] skill 将本地报告设为唯一可编辑来源，并通过逐文件 rsync 与 SHA-256
      校验同步报告和索引到 AutoDL。

## Definition of Done (team quality bar)

* 项目级 skill 的职责、边界和使用方式经用户确认。
* 报告方案不要求修改训练、推理或评估接口。
* skill、模板、报告契约和前向验证结果均完成。
* 文档与任务记录反映最终决策。

## Out of Scope (explicit)

* 实现新的实验追踪服务或数据库。
* 修改模型训练、推理、评估逻辑。
* 重写或迁移既有报告。
* 自动修改论文结论、主线 PRD、fallback 门槛或实验协议。
* 首版自动生成跨 seed 统计、论文表格或图形。

## Technical Notes

* 当前任务目录：`.trellis/tasks/07-14-experiment-reporting-skill/`。
* 主线 `info.md` 已冻结 run ID、manifest 字段和稳定运行产物布局。
* 主线文档所有权约定由父任务负责研究声明、子任务负责各自实验执行，适合采用“子任务报告、父任务索引”。
* 前向验证覆盖一个 manifest 完整的 synthetic 成功 fixture，以及一个缺少 manifest 的真实历史 Adapter 结果目录。
* Spec update review: no `.trellis/spec/` change is needed. The project-specific reporting contract is intentionally owned by this local skill, and duplicating it in backend code specs would create two sources of truth.
