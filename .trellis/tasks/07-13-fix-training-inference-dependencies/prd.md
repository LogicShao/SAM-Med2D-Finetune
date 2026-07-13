# 修复训练与推理模块重复依赖

## Goal

消除训练与推理路径中已确认的重复 checkpoint 加载、布尔 CLI 解析错误、BraTS 类别常量分散和批量验证对推理脚本的反向依赖，同时为抽取后的行为补充回归测试。

## Requirements

* `model_factory` 成为多任务模型 checkpoint 加载和输入通道适配的唯一入口，不静默忽略 checkpoint 加载失败。
* `train_multitask.py` 使用 `model_factory`，不再重复加载 checkpoint 或替换 `patch_embed`。
* 三个训练/评估入口使用可靠的字符串布尔解析。
* 定义共享 BraTS 类别常量，并保持预测级联顺序与展示/评估顺序的不同语义。
* 批量验证不再从 `infer_volume.py` 导入常量或通用工具。
* 为模型工厂、三维后处理和提示策略添加定向单元测试。

## Acceptance Criteria

* [x] 多任务训练与推理各只经过一条明确的 checkpoint 加载路径。
* [x] `--encoder_adapter false` 被解析为 `False`。
* [x] BraTS 类别展示顺序只维护一处，预测级联顺序从共享契约派生或显式引用。
* [x] 批量验证不依赖 `infer_volume.py` 的常量、IO 或 CLI 工具。
* [x] 新增测试覆盖所改行为，且相关测试通过。

## Out of Scope

* 不重写完整的 `infer_volume.py` 推理引擎。
* 不调整训练损失、数据协议、提示预设或后处理算法参数。
* 不安装、升级或移除依赖。

## Technical Approach

新增轻量模块承载 BraTS 契约、CLI 解析和推理输出/元数据工具。模型工厂通过未加载 checkpoint 的模型构建入口加载一次基础权重，再完成输入通道适配；训练脚本复用该入口。

## Technical Notes

* `segment_anything/build_sam.py` 的注册器会按 `args.sam_checkpoint` 自动加载权重，因此不能在已传入 checkpoint 的情况下再次显式加载。
* 工作区已有用户未提交修改；仅编辑本任务范围内的文件，不回退其他变更。
