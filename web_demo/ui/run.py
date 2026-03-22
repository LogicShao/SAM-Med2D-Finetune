from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from web_demo.services.job_state import get_job
from web_demo.services.logger import read_log_tail
from web_demo.services.pipeline import prepare_case_input, start_pipeline_job
from web_demo.ui.common import build_notice, build_page_context


router = APIRouter()


def _render_run_template(
    request: Request,
    *,
    title: str,
    active_nav: str,
    **context: object,
) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="run.html",
        context=build_page_context(request, title=title, active_nav=active_nav, **context),
    )


def _build_job_status_payload(run_id: str) -> dict[str, object]:
    job = get_job(run_id)
    return {
        **job,
        "log_tail": read_log_tail(job.get("log_path"), max_lines=40),
    }


@router.get("/run", response_class=HTMLResponse)
def run_page(request: Request) -> HTMLResponse:
    return _render_run_template(
        request,
        title="上传运行",
        active_nav="run",
        page_mode="form",
        notice=None,
    )


@router.get("/run/{run_id}", response_class=HTMLResponse)
def run_wait_page(request: Request, run_id: str) -> HTMLResponse:
    try:
        payload = _build_job_status_payload(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未找到运行任务: {run_id}") from exc

    return _render_run_template(
        request,
        title=f"运行中 - {run_id}",
        active_nav="run",
        page_mode="wait",
        run_id=run_id,
        initial_state=payload,
        initial_state_json=json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"),
    )


@router.get("/api/runs/{run_id}", response_class=JSONResponse)
def run_status_api(run_id: str) -> JSONResponse:
    try:
        payload = _build_job_status_payload(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未找到运行任务: {run_id}") from exc
    return JSONResponse(payload)


@router.post("/run")
def run_case(
    request: Request,
    case_dir: str = Form(default=""),
    files: list[UploadFile] | None = File(default=None),
):
    try:
        case_input = prepare_case_input(case_dir_text=case_dir, uploaded_files=files)
        job = start_pipeline_job(case_input)
        return RedirectResponse(url=f"/run/{job['run_id']}", status_code=303)
    except Exception as exc:
        return _render_run_template(
            request,
            title="上传运行",
            active_nav="run",
            page_mode="form",
            notice=build_notice(str(exc), tone="error"),
        )