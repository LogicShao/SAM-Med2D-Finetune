from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from web_demo.services.pipeline import prepare_case_input, run_full_pipeline
from web_demo.ui.common import render_notice, render_page


router = APIRouter()


def _render_run_form(error_message: str | None = None) -> HTMLResponse:
    notice_html = render_notice(error_message, tone="error") if error_message else ""
    body = f"""
    <section class="section-heading page-header">
      <div>
        <div class="eyebrow">Single Case Demo</div>
        <h1>上传病例并运行</h1>
        <p>当前仅支持单病例串行处理。页面会复用现有推理脚本、后处理逻辑和 3D HTML 生成脚本。</p>
      </div>
      <a class="button ghost" href="/">返回首页</a>
    </section>

    {notice_html}

    <section class="card form-card">
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


import json as json_lib

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from web_demo.services.job_state import get_job
from web_demo.services.logger import read_log_tail
from web_demo.services.pipeline import start_pipeline_job


def _job_status_payload(run_id: str) -> dict[str, object]:
    job = get_job(run_id)
    return {
        **job,
        "log_tail": read_log_tail(job.get("log_path"), max_lines=40),
    }


def _render_run_form(error_message: str | None = None) -> HTMLResponse:
    notice_html = render_notice(error_message, tone="error") if error_message else ""
    body = f"""
    <section class="section-heading page-header">
      <div>
        <div class="eyebrow">Single Case Demo</div>
        <h1>上传病例并运行</h1>
        <p>点击开始处理后，页面会切换到等待态，并按阶段展示当前处理步骤与后台日志。</p>
      </div>
      <a class="button ghost" href="/">返回首页</a>
    </section>

    {notice_html}

    <section class="card form-card">
      <form class="upload-form" id="run-form" action="/run" method="post" enctype="multipart/form-data">
        <div class="form-grid">
          <label class="field">
            <span>本机病例目录</span>
            <input type="text" name="case_dir" placeholder="例如：D:/proj/.../BraTS2021_00526">
            <small>优先支持单病例目录，目录内应包含 t1 / t1ce / t2 / flair 四个模态。</small>
          </label>

          <label class="field">
            <span>上传病例文件</span>
            <input type="file" name="files" multiple accept=".nii,.nii.gz">
            <small>可直接上传单病例所需 NIfTI 文件，后端会先落盘，再进入串行处理流程。</small>
          </label>

          <label class="field">
            <span>上传病例目录</span>
            <input type="file" name="files" multiple webkitdirectory directory>
            <small>兼容 Chromium 目录上传；当前仍按单病例串行 demo 执行。</small>
          </label>
        </div>

        <div class="inline-note">
          <strong>说明：</strong> 当前实现优先补齐可轮询的等待页、阶段状态与日志 tail，不引入 websocket、celery 或复杂队列。
        </div>

        <div class="hero-actions">
          <button class="button primary" id="run-submit" type="submit">开始处理</button>
          <a class="button secondary" href="/samples">先看样例结果</a>
        </div>
      </form>
    </section>

    <script>
      (() => {{
        const form = document.getElementById("run-form");
        const submitButton = document.getElementById("run-submit");
        if (!form || !submitButton) {{
          return;
        }}
        form.addEventListener("submit", () => {{
          submitButton.disabled = true;
          submitButton.textContent = "正在接收病例...";
          submitButton.classList.add("is-disabled");
        }}, {{ once: true }});
      }})();
    </script>
    """
    return render_page(title="上传运行", body=body, active_nav="run")


def _render_run_wait_page(run_id: str) -> HTMLResponse:
    try:
        payload = _job_status_payload(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未找到运行任务: {run_id}") from exc

    initial_json = json_lib.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    body = f"""
    <section class="section-heading page-header">
      <div>
        <div class="eyebrow">Pipeline Running</div>
        <h1>正在处理病例</h1>
        <p>页面会定时轮询当前任务状态，并展示最近日志。任务完成后会自动提供结果入口。</p>
      </div>
      <div class="hero-actions">
        <a class="button ghost" href="/run">返回上传页</a>
        <a class="button secondary" href="/samples">查看样例</a>
      </div>
    </section>

    <section class="card job-overview-card">
      <div class="job-status-header">
        <div class="job-status-chip is-running" id="job-status-chip">处理中</div>
        <div class="job-status-copy">
          <h2 id="job-current-stage">已接收病例</h2>
          <p id="job-message">病例文件已接收，等待进入自动分割。</p>
        </div>
      </div>
      <div class="job-meta-grid">
        <article class="metric-card job-meta-card">
          <span>病例 ID</span>
          <strong id="job-case-id">{payload["case_id"]}</strong>
        </article>
        <article class="metric-card job-meta-card">
          <span>Run ID</span>
          <strong id="job-run-id">{payload["run_id"]}</strong>
        </article>
        <article class="metric-card job-meta-card">
          <span>来源</span>
          <strong id="job-source-type">{payload["source_type"]}</strong>
        </article>
        <article class="metric-card job-meta-card">
          <span>结果目录</span>
          <strong id="job-result-dir">{payload["result_dir"]}</strong>
        </article>
      </div>
    </section>

    <section class="card">
      <div class="section-heading compact">
        <div>
          <h2>阶段状态</h2>
          <p>当前为串行 demo，不展示百分比，只展示明确阶段切换。</p>
        </div>
      </div>
      <div class="job-stage-grid" id="job-stage-grid"></div>
    </section>

    <section class="card">
      <div class="section-heading compact">
        <div>
          <h2>后台日志</h2>
          <p>展示最近 40 行日志，便于判断当前执行到哪一步。</p>
        </div>
      </div>
      <pre class="job-log-panel" id="job-log-panel">等待日志输出...</pre>
      <div class="hero-actions job-action-row">
        <a class="button primary" id="job-result-link" href="#" hidden>查看结果</a>
        <span class="job-redirect-note" id="job-redirect-note" hidden>处理完成，正在跳转结果页...</span>
      </div>
    </section>

    <script>
      (() => {{
        const runId = {json_lib.dumps(run_id, ensure_ascii=False)};
        const statusUrl = `/api/runs/${{runId}}`;
        const pollIntervalMs = 1500;
        let redirectStarted = false;

        const initialState = {initial_json};
        const chip = document.getElementById("job-status-chip");
        const currentStage = document.getElementById("job-current-stage");
        const message = document.getElementById("job-message");
        const stageGrid = document.getElementById("job-stage-grid");
        const logPanel = document.getElementById("job-log-panel");
        const resultLink = document.getElementById("job-result-link");
        const redirectNote = document.getElementById("job-redirect-note");

        const escapeHtml = (value) => String(value ?? "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");

        const statusTextMap = {{
          running: "处理中",
          success: "已完成",
          failed: "失败",
          pending: "等待中",
        }};

        function renderStages(stages) {{
          stageGrid.innerHTML = (stages || []).map((stage) => `
            <article class="job-stage-card is-${{escapeHtml(stage.state)}}">
              <div class="job-stage-top">
                <span class="job-stage-label">${{escapeHtml(stage.label)}}</span>
                <span class="job-stage-state">${{escapeHtml(statusTextMap[stage.state] || stage.state)}}</span>
              </div>
              <p>${{escapeHtml(stage.message || "等待执行")}}</p>
            </article>
          `).join("");
        }}

        function renderLogs(lines) {{
          const text = (lines || []).length ? lines.join("\\n") : "等待日志输出...";
          logPanel.textContent = text;
          logPanel.scrollTop = logPanel.scrollHeight;
        }}

        function renderState(data) {{
          chip.textContent = statusTextMap[data.status] || data.status || "处理中";
          chip.className = `job-status-chip is-${{data.status || "running"}}`;
          currentStage.textContent = data.current_stage_label || data.current_stage || "处理中";
          message.textContent = data.message || "后台正在执行，请稍候。";
          renderStages(data.stages);
          renderLogs(data.log_tail);

          if (data.result_url) {{
            resultLink.hidden = false;
            resultLink.href = data.result_url;
          }}

          if (data.status === "success" && data.result_url && !redirectStarted) {{
            redirectStarted = true;
            redirectNote.hidden = false;
            window.setTimeout(() => {{
              window.location.href = data.result_url;
            }}, 1200);
          }}
        }}

        async function poll() {{
          try {{
            const response = await fetch(statusUrl, {{ cache: "no-store" }});
            if (!response.ok) {{
              throw new Error(`HTTP ${{response.status}}`);
            }}
            const data = await response.json();
            renderState(data);
            if (data.status === "running" || data.status === "pending") {{
              window.setTimeout(poll, pollIntervalMs);
            }}
          }} catch (error) {{
            message.textContent = `状态轮询失败：${{error.message}}`;
            window.setTimeout(poll, pollIntervalMs);
          }}
        }}

        renderState(initialState);
        if (initialState.status === "running" || initialState.status === "pending") {{
          window.setTimeout(poll, pollIntervalMs);
        }}
      }})();
    </script>
    """
    return render_page(title=f"运行中 - {run_id}", body=body, active_nav="run")


router = APIRouter()


@router.get("/run", response_class=HTMLResponse)
def run_page() -> HTMLResponse:
    return _render_run_form()


@router.get("/run/{run_id}", response_class=HTMLResponse)
def run_wait_page(run_id: str) -> HTMLResponse:
    return _render_run_wait_page(run_id)


@router.get("/api/runs/{run_id}", response_class=JSONResponse)
def run_status_api(run_id: str) -> JSONResponse:
    try:
        payload = _job_status_payload(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未找到运行任务: {run_id}") from exc
    return JSONResponse(payload)


@router.post("/run")
def run_case(
    case_dir: str = Form(default=""),
    files: list[UploadFile] | None = File(default=None),
):
    try:
        case_input = prepare_case_input(case_dir_text=case_dir, uploaded_files=files)
        job = start_pipeline_job(case_input)
        return RedirectResponse(url=f"/run/{job['run_id']}", status_code=303)
    except Exception as exc:
        return _render_run_form(error_message=str(exc))
