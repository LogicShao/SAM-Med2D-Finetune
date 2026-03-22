# SAM-Med2D 最小演示版

这个目录只保留一条可演示主链路：

1. 选择已有病例结果
2. 或上传单病例并串行运行
3. 自动分割
4. 后处理 / 3D 重建
5. 结果查看

## 目录结构

```text
web_demo/
├── app.py
├── config.py
├── README.md
├── services/
│   ├── cases.py
│   ├── pipeline.py
│   └── results.py
├── static/
│   └── style.css
├── templates/
└── ui/
    ├── common.py
    ├── home.py
    ├── result.py
    └── run.py
```

## 设计取向

- `app.py` 只负责应用创建、静态资源挂载和路由注册。
- `services/` 负责样例发现、流程封装和结果读取。
- `ui/` 只负责页面与路由。
- `static/style.css` 承担全部样式。
- 优先复用 `infer_volume.py`、`postprocess_3d.py`、`visualize_case.py` 和已有 `outputs/`。

## 依赖

如果只补 Web demo 相关依赖，至少安装：

```bash
pip install fastapi uvicorn python-multipart
```

或者直接：

```bash
pip install -r requirements.txt
```

## 默认模型配置

当前 Web demo 默认整病例推理主模型为：

- checkpoint：`workdir_multi_task/models/finetune_no_stop_lora/lora_adapters`
- `finetune_method`：`lora`

之所以暂时继续保留这套配置，是为了与既有整病例回归结果和展示样例保持一致；后续可在补齐 `Adapter` 版正式回归后再统一切换默认 pipeline。

## 启动

从仓库根目录执行：

```bash
C:/Users/acs/.conda/envs/Brain-Tumor-Segmentation/python.exe -m web_demo.app
```

默认地址：

```text
http://127.0.0.1:7860
```

也可以使用：

```bash
C:/Users/acs/.conda/envs/Brain-Tumor-Segmentation/python.exe -m uvicorn web_demo.app:app --host 127.0.0.1 --port 7860
```

## 演示方式

### 1. 样例优先

- 进入首页后点击“选择样例病例”。
- 页面会从已有 `outputs/` 中筛出 3 到 6 个稳定病例。
- 这些病例直接复用已有 `case_meta.json`、`preview_3d*.html` 和结果目录。
- 这条链路不重新计算，稳定性最高，适合答辩和截图。

### 2. 上传并运行

上传页支持两种输入方式：

- 直接填写本机病例目录路径
- 上传单病例所需的 NIfTI 文件，或用浏览器目录上传

后端按单病例串行执行：

1. `infer_volume.py`
2. `postprocess_3d.py` 中的后处理逻辑
3. `visualize_case.py`

结果统一写到：

```text
outputs/web_demo_runs/<timestamp>_<case_id>/
```

## 结果页保留内容

- 病例信息
- 处理状态
- 3D HTML 结果展示
- 2D 切片与叠加图
- 简短结果说明

默认不展示：

- 训练曲线
- 大段 Dice 表格
- 多实验科研看板
- 无关控制按钮

## 当前限制

- 上传链路只面向单病例串行 demo，不做任务队列。
- 目录上传主要依赖 Chromium 系浏览器的 `webkitdirectory`。
- 样例结果页优先读取已有 `outputs/`；如果缺少 2D 图，会按需从 NIfTI 生成简易叠加图。
- 当前实现采用 `FastAPI` 页面层，未额外依赖 `Gradio` 才能跑通主演示流程。
