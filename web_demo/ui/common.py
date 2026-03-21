from __future__ import annotations

from html import escape

from fastapi.responses import HTMLResponse

from web_demo.config import APP_DESCRIPTION, APP_NAME


def render_page(title: str, body: str, active_nav: str) -> HTMLResponse:
    nav_items = (
        ("home", "/", "首页"),
        ("samples", "/samples", "样例病例"),
        ("run", "/run", "上传运行"),
    )
    nav_html = "".join(
        f"<a class=\"nav-link{' active' if key == active_nav else ''}\" href=\"{href}\">{escape(label)}</a>"
        for key, href, label in nav_items
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} - {escape(APP_NAME)}</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand-block">
        <a class="brand-link" href="/">{escape(APP_NAME)}</a>
        <p class="brand-copy">{escape(APP_DESCRIPTION)}</p>
      </div>
      <nav class="nav-links">{nav_html}</nav>
    </div>
  </header>
  <main class="page-shell">
    {body}
  </main>
  <footer class="footer">
    <span>最小演示版仅保留单病例流程，不包含任务队列、数据库和科研看板。</span>
  </footer>
</body>
</html>"""
    return HTMLResponse(html)


def render_notice(message: str, tone: str = "info") -> str:
    return f"<div class=\"notice {escape(tone)}\">{escape(message)}</div>"
