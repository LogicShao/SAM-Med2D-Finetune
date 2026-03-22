from __future__ import annotations

from typing import Any

from fastapi import Request

from web_demo.config import APP_DESCRIPTION, APP_NAME


NAV_ITEMS = (
    {"key": "home", "href": "/", "label": "首页"},
    {"key": "samples", "href": "/samples", "label": "样例病例"},
    {"key": "run", "href": "/run", "label": "上传运行"},
)

FOOTER_TEXT = "最小演示版仅保留单病例流程，不包含任务队列、数据库和科研看板。"


def build_page_context(
    request: Request,
    *,
    title: str,
    active_nav: str,
    **context: Any,
) -> dict[str, Any]:
    return {
        "request": request,
        "page_title": title,
        "active_nav": active_nav,
        "app_name": APP_NAME,
        "app_description": APP_DESCRIPTION,
        "nav_items": NAV_ITEMS,
        "footer_text": FOOTER_TEXT,
        **context,
    }


def build_notice(message: str | None, tone: str = "info") -> dict[str, str] | None:
    if not message:
        return None
    return {"message": message, "tone": tone}