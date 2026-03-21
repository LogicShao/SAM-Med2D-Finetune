from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from web_demo.services.pipeline import prepare_case_input, run_full_pipeline
from web_demo.ui.common import render_notice, render_page


router = APIRouter()


def _render_run_form(error_message: str | None = None) -> HTMLResponse:
    notice_html = render_notice(error_message, tone="error") if error_message else ""
    body = f"""
    <section class="section-heading">
      <div>
        <div class="eyebrow">Single Case Demo</div>
        <h1>上传病例并运行</h1>
        <p>当前仅支持单病例串行处理。页面会复用现有推理脚本、后处理逻辑和 3D HTML 生成脚本。</p>
      </div>
      <a class="button ghost" href="/">返回首页</a>
    </section>

    {notice_html}

    <section class="card">
      <form class="upload-form" action="/run" method="post" enctype="multipart/form-data">
        <div class="form-grid">
          <label class="field">
            <span>本机病例目录</span>
            <input type="text" name="case_dir" placeholder="例如：D:/proj/.../BraTS2021_00526">
            <small>如果你在本机直接部署 demo，优先填写病例目录路径，稳定性最高。</small>
          </label>

          <label class="field">
            <span>上传病例文件</span>
            <input type="file" name="files" multiple accept=".nii,.nii.gz">
            <small>上传单病例所需的 4 个模态文件，可选附带 seg 文件。</small>
          </label>

          <label class="field">
            <span>上传病例目录</span>
            <input type="file" name="files" multiple webkitdirectory directory>
            <small>兼容 Chromium 系浏览器的目录上传；后端仍按单病例串行执行。</small>
          </label>
        </div>

        <div class="inline-note">
          <strong>真实状态说明：</strong> 上传链路当前是单病例串行 demo，不做任务队列和并发；如果模型权重或 GPU 环境不可用，页面会返回真实错误信息。
        </div>

        <div class="hero-actions">
          <button class="button primary" type="submit">开始处理</button>
          <a class="button secondary" href="/samples">先看样例结果</a>
        </div>
      </form>
    </section>
    """
    return render_page(title="上传运行", body=body, active_nav="run")


@router.get("/run", response_class=HTMLResponse)
def run_page() -> HTMLResponse:
    return _render_run_form()


@router.post("/run")
def run_case(
    case_dir: str = Form(default=""),
    files: list[UploadFile] | None = File(default=None),
):
    try:
        case_input = prepare_case_input(case_dir_text=case_dir, uploaded_files=files)
        pipeline_result = run_full_pipeline(case_input)
        return RedirectResponse(url=f"/results/{pipeline_result['result_id']}", status_code=303)
    except Exception as exc:
        return _render_run_form(error_message=str(exc))
