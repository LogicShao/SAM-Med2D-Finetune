# Current Todo

本文件用于约束当前开发周期内的实验与代码变更范围。后续对话、脚本实现与结果解读默认参考本文件；当本轮开发完成并确认不再需要该临时计划时，应删除本文件。

## 目标

围绕当前 `YOLO -> SAM -> 3D postprocess` 端到端链路，优先解决困难病例的漏检、提示不稳定和 3D 连续性不足问题。

当前固定基线：

- YOLO 检测器：`workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt`
- 默认工作点：`conf=0.05`
- Prompt 策略：`top-1 bbox`
- 后处理：`closing_radius=2`, `opening_radius=1`, `wt_keep_largest=true`, `keep_topk_tc=1`, `keep_topk_et=1`, `z_smooth_iterations=3`

## 优先级

1. 困难病例上扫 YOLO `conf/iou`
2. 做 `top-1` vs `top-2 + 规则过滤` 对照
3. 对低分病例扫后处理参数

## 阶段 1：困难病例上扫 YOLO `conf/iou`

目标：先锁定 detector 工作点，再动 prompt 策略；本阶段只回答“`yolo_box` 应该用哪个 `conf/iou`”。

范围约束：

- 只改 YOLO 推理阈值，不改 YOLO 权重、`imgsz`、SAM 权重、prompt 合并逻辑和 3D 后处理规则
- 困难病例作为主评估集，全量验证集只做最终回归确认
- 阶段一结束前，不进入 `top-2` prompt 或后处理参数扫描

前置实现缺口：

- `tools/evaluate_yolo_recall.py` 已支持同时扫描 `conf/iou`
- `sam_med2d_finetune.inference.volume` / `sam_med2d_finetune.inference.batch_validate` 目前只暴露 `--yolo_conf`
- `YoloBoxPromptProvider` 当前将 NMS `iou` 写死为 `0.60`
- 因此阶段一的第一个交付不是跑实验，而是先补齐端到端链路中的 `--yolo_iou` 透传、记录与落盘

固定基线：

- YOLO：`workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt`
- Prompt：`yolo_box + top-1 bbox`
- 后处理：`closing_radius=2`, `opening_radius=1`, `wt_keep_largest=true`, `keep_topk_tc=1`, `keep_topk_et=1`, `z_smooth_iterations=3`

参数网格：

- `conf`: `0.03`, `0.05`, `0.08`, `0.10`, `0.15`
- `iou`: `0.50`, `0.60`, `0.70`

困难病例集合冻结规则：

- 以当前默认工作点 `conf=0.05, iou=0.60` 的端到端结果为基线
- 从 `summary_metrics.csv` 中选 `post_mean_dice` 最低的病例作为困难病例主集
- 若某病例出现 `ET` 或 `TC` Dice 接近失效，也强制纳入，即使其 `post_mean_dice` 不是最低
- 困难病例列表一旦确定，阶段一期间保持不变，避免边跑边换评估集

执行步骤：

1. 补齐参数透传
   - 为 `sam_med2d_finetune.inference.volume`、`sam_med2d_finetune.inference.batch_validate` 增加 `--yolo_iou`
   - 将 `case_meta.json`、`summary_metrics.json`、`summary.md` 记录 `yolo_iou`
   - 保持默认值仍为 `0.60`，确保现有基线可复现
2. 做 YOLO 层粗筛
   - 用 `tools/evaluate_yolo_recall.py` 在固定数据集上扫完整 `conf/iou` 网格
   - 先只看检测层指标，筛出 4 到 6 个候选工作点
   - 淘汰明显漏层严重或背景误报过高的组合，减少后续端到端成本
3. 做困难病例端到端验证
   - 对候选工作点运行 `sam_med2d_finetune.inference.batch_validate`
   - 输出每病例 `case_meta.json`、`preview_3d_compare_all.html`、汇总指标文件
   - 重点比较 `ET/TC/WT Dice`、`Mean Dice`、`post Mean Dice`
4. 做全量回归确认
   - 仅对前 2 个候选工作点补跑全量验证集
   - 防止“困难病例收益”来自对小集合过拟合
5. 选定阶段一工作点
   - 产出一个推荐值和一个保守备选值
   - 在文档中明确为什么选它，而不是只贴分数

建议筛选门槛：

- 第一层：优先保证 `slice_recall_any_box`
- 第二层：`slice_recall_iou_0.30` 不能明显退化
- 第三层：困难病例 `post Mean Dice` 必须不低于当前基线
- 若多个候选接近，优先选 `background_false_positive_rate` 更低、`avg_boxes_per_positive_slice` 更稳定的设置

输出目录规范：

- YOLO 粗筛输出到 `outputs/stage1_yolo_recall_scan/`
- 端到端验证输出到 `outputs/stage1_e2e/conf_<x>_iou_<y>/`
- 所有对比表最终汇总到 `outputs/stage1_summary/`

阶段一推荐产物：

- `hard_cases.txt`：冻结后的困难病例列表
- `recall_scan.csv`：15 个工作点的 YOLO 层汇总表
- `candidate_shortlist.md`：进入端到端验证的候选说明
- `summary_metrics.json`
- `summary_metrics.csv`
- `summary.md`
- 每病例 `case_meta.json`
- 每病例 `preview_3d_compare_all.html`

命令模板：

```bash
PYTHONPATH=src python -m sam_med2d_finetune.tools.evaluate_yolo_recall \
  --model workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt \
  --data datasets/brats_yolo_dev \
  --split val \
  --conf_values 0.03,0.05,0.08,0.10,0.15 \
  --iou 0.60 \
  --out_dir outputs/stage1_yolo_recall_scan/iou_0p60
```

```bash
PYTHONPATH=src python -m sam_med2d_finetune.inference.batch_validate \
  --cases_root <cases_root> \
  --case_ids <hard_case_1> <hard_case_2> \
  --output_root outputs/stage1_e2e/conf_0p05_iou_0p60 \
  --sam_checkpoint <sam_checkpoint> \
  --finetuned_checkpoint <finetuned_checkpoint> \
  --finetune_method lora \
  --prompt_mode yolo_box \
  --yolo_checkpoint workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt \
  --yolo_conf 0.05 \
  --yolo_iou 0.60 \
  --postprocess true \
  --closing_radius 2 \
  --opening_radius 1 \
  --wt_keep_largest true \
  --keep_topk_tc 1 \
  --keep_topk_et 1 \
  --z_smooth_iterations 3
```

完成标准：

- 已能在端到端链路中真正扫描 `conf/iou`，而不是只扫描 `conf`
- 有冻结的困难病例主集和可复现的基线快照
- 至少得到一个推荐工作点和一个备选工作点
- 推荐工作点在困难病例上的 `post Mean Dice` 稳定优于或不低于基线
- 推荐工作点在全量验证集上没有明显回退

## 阶段 2：`top-1` vs `top-2 + 规则过滤`

目标：验证单框 prompt 是否限制了困难病例表现。

第一版实现方式：

- 默认保留 `top-1`
- 对照组取 `top-2`
- 若第二个框满足规则，则与第一框合并为一个外接框，再送入 SAM

建议过滤规则：

- `score2 >= 0.5 * score1`
- `area2 >= 0.1 * area1`
- `IoU(box1, box2) < 0.9`
- `area2 <= 2.0 * area1`

比较重点：

- 困难病例 `post Mean Dice`
- `ET/TC` 是否改善
- 3D 预览中是否引入更多漂浮假阳性

## 阶段 3：低分病例后处理参数扫描

目标：进一步修复层间断裂、噪点和 3D 不连续。

建议参数网格：

- `closing_radius`: `1`, `2`, `3`
- `z_smooth_iterations`: `1`, `2`, `3`
- `keep_topk_tc`: `1`, `2`
- `keep_topk_et`: `1`, `2`

固定不变：

- YOLO 模型与工作点
- Prompt 策略
- SAM 微调权重

观察重点：

- `WT/TC` 是否继续提升
- `ET` 是否被过度抹掉
- 3D HTML 中漂浮块是否减少
- 层间连续性是否增强

## 困难病例定义

优先采用端到端定义：

- `post Mean Dice` 最低的病例
- 或 `ET/TC` 明显失效的病例

困难病例用于误差分析与工作点筛选，不用于单独“拟合”或作为最终结论依据。

## 输出规范

每轮实验尽量统一输出：

- `summary_metrics.json`
- `summary_metrics.csv`
- `summary.md`
- 每病例 `case_meta.json`
- 每病例 `preview_3d_compare_all.html`

## 完成标准

- 困难病例平均 `post Mean Dice` 有稳定提升
- 整体验证集结果不明显退步
- `ET/TC` 不因后处理而显著塌缩
- 3D 结构质量主观上更稳定

## 文档生命周期

- 本文件仅服务当前开发周期
- 本轮任务收尾后，应删除 `todo.md`
- README 中只保留已经验证完成、适合长期保留的结论，不直接复制本文件中的临时实验计划
