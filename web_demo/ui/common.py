from __future__ import annotations

from typing import Any

from fastapi import Request

from web_demo.config import APP_DESCRIPTION, APP_NAME


NAV_ITEMS = (
    {"key": "home", "href": "/", "label": "首页"},
    {"key": "samples", "href": "/samples", "label": "病例列表"},
    {"key": "run", "href": "/run", "label": "病例处理"},
)

FOOTER_TEXT = "本系统用于影像处理流程展示，结果仅供辅助分析参考。"


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