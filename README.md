# SAM-Med2D Finetune

这个仓库现在只保留一件事：给后续 session 和开发者提供清晰的代码入口。

旧版长文档已经归档到 `docs/archive/readme-full-2026-07-14.md`，根目录 README 只保留当前活跃代码、启动方式和最常用命令。

## 代码入口

- 训练入口：`PYTHONPATH=src python -m sam_med2d_finetune.training.train_multitask`
- 单病例推理：`PYTHONPATH=src python -m sam_med2d_finetune.inference.volume`
- 批量验证：`PYTHONPATH=src python -m sam_med2d_finetune.inference.batch_validate`
- Web Demo：`PYTHONPATH=src python -m sam_med2d_finetune.web_demo.app`
- 固定实验脚本：`finetune_scripts/multi_task/adapter.py`、`finetune_scripts/multi_task/lora.py`

## 活跃目录

```text
src/
├── sam_med2d_finetune/
│   ├── brats/          # BraTS 病例、cache、指标契约
│   ├── training/       # 多任务训练主链路
│   ├── inference/      # 整病例推理、后处理、可视化
│   ├── models/         # 模型构建与权重加载
│   ├── tools/          # 数据划分、cache、辅助工具
│   ├── utils/          # CLI 和训练通用逻辑
│   └── web_demo/       # Web 展示和单病例处理入口
└── segment_anything/   # vendored SAM-Med2D 实现
```

## 快速开始

先安装依赖：

```bash
pip install -r requirements.txt
```

再从仓库根目录执行：

```bash
PYTHONPATH=src python -m sam_med2d_finetune.training.train_multitask \
  --finetune_method adapter \
  --train_data_path data_brats_raw/train \
  --val_data_path data_brats_raw/val \
  --work_dir workdir_multi_task
```

```bash
PYTHONPATH=src python -m sam_med2d_finetune.inference.batch_validate \
  --cases_root data_brats_raw/val \
  --output_root outputs/validation_run \
  --sam_checkpoint pretrain_model/sam-med2d_b.pth \
  --finetuned_checkpoint workdir_multi_task/models/finetune_adapter/best_model.pth \
  --finetune_method adapter
```

## 约定

- 不再新增根目录 Python 入口脚本，统一走 `PYTHONPATH=src python -m ...`
- 当前只维护多任务 BraTS 主链路，历史单任务代码已移除
- `report/` 是本地私有材料目录，已从 Git 跟踪中移除，不作为仓库文档入口
- 结果、日志、权重和大图片不要放进 Git

## 进一步说明

- 目录规范：`.trellis/spec/backend/directory-structure.md`
- 质量规范：`.trellis/spec/backend/quality-guidelines.md`
- Web Demo 说明：`src/sam_med2d_finetune/web_demo/README.md`
