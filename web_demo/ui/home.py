from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from web_demo.services.cases import list_sample_cases
from web_demo.ui.common import build_page_context


router = APIRouter()

HOME_STEPS = (
    "选择已完成处理的病例，或导入单病例目录/文件。",
    "系统完成自动分割处理。",
    "系统生成处理结果与三维可视化内容。",
    "在结果页查看病例信息、处理状态、三维结果与关键切片。",
)

HOME_CONSTRAINTS = (
    "当前支持单病例处理与结果查看。",
    "可直接查看已完成处理的病例结果。",
    "系统提供自动分割、结果处理与三维可视化。",
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
        title="病例列表",
        active_nav="samples",
        page_mode="samples",
        sample_cases=list_sample_cases(),
    )
