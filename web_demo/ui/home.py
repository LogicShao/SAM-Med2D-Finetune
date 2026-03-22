from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from web_demo.services.cases import list_sample_cases
from web_demo.ui.common import build_page_context


router = APIRouter()

HOME_STEPS = (
    "选择已有病例结果，或上传单病例目录/文件。",
    "调用现有整病例推理脚本完成自动分割。",
    "调用现有后处理和 3D 可视化逻辑生成结果目录。",
    "在结果页查看病例信息、处理状态、3D 结果和 2D 切片。",
)

HOME_CONSTRAINTS = (
    "当前仅保证单病例串行 demo，不做任务队列、数据库和并发管理。",
    "优先保证“选择样例病例 -> 查看结果”稳定可演示。",
    "上传链路复用现有脚本，不额外改写算法流程。",
)


def _render_home_template(request: Request, *, title: str, active_nav: str, **context: object) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="home.html",
        context=build_page_context(request, title=title, active_nav=active_nav, **context),
    )


@router.get("/", response_class=HTMLResponse)
def home_page(request: Request) -> HTMLResponse:
    sample_cases = list_sample_cases()
    return _render_home_template(
        request,
        title="首页",
        active_nav="home",
        page_mode="home",
        sample_count=len(sample_cases),
        steps=HOME_STEPS,
        constraints=HOME_CONSTRAINTS,
    )


@router.get("/samples", response_class=HTMLResponse)
def sample_cases_page(request: Request) -> HTMLResponse:
    return _render_home_template(
        request,
        title="样例病例",
        active_nav="samples",
        page_mode="samples",
        sample_cases=list_sample_cases(),
    )