from __future__ import annotations

from html import escape

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from web_demo.services.results import get_viewer_file_for_result, load_result_view
from web_demo.ui.common import render_page


router = APIRouter()


def _render_case_info_rows(items: list[dict[str, str]]) -> str:
    return "".join(
        f"<div class=\"info-row\"><span class=\"info-label\">{escape(item['label'])}</span>"
        f"<span class=\"info-value\">{escape(item['value'])}</span></div>"
        for item in items
    )


@router.get("/results/{result_id}", response_class=HTMLResponse)
def result_page(result_id: str) -> HTMLResponse:
    try:
        result = load_result_view(result_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    metric_badges_html = "".join(
        f"<div class=\"metric-badge metric-card\"><span>{escape(item['label'])}</span><strong>{escape(item['value'])}</strong></div>"
        for item in result["metric_badges"]
    )
    status_cards_html = "".join(
        f"""
        <article class="status-card {escape(card['state'])}">
          <h3>{escape(card['label'])}</h3>
          <p>{escape(card['detail'])}</p>
        </article>
        """
        for card in result["status_cards"]
    )
    summary_html = "".join(f"<p>{escape(line)}</p>" for line in result["summary_lines"])
    viewer_html = (
        f"<iframe class=\"viewer-frame\" src=\"{escape(result['viewer_url'])}\" loading=\"lazy\"></iframe>"
        if result["viewer_url"]
        else """
        <div class="empty-state compact">
          <h3>未找到 3D HTML 预览</h3>
          <p>当前结果目录中没有可嵌入的 viewer.html 或 preview_3d*.html。</p>
        </div>
        """
    )

    if result["slice_images"]:
        slice_gallery_html = "".join(
            f"""
            <figure class="slice-card">
              <img src="{escape(item['url'])}" alt="{escape(item['label'])}">
              <figcaption>{escape(item['label'])}</figcaption>
            </figure>
            """
            for item in result["slice_images"]
        )
    else:
        slice_gallery_html = """
        <div class="empty-state compact">
          <h3>暂无 2D 切片图</h3>
          <p>当前结果目录缺少可复用切片图，且未能按需生成叠加图。</p>
        </div>
        """

    body = f"""
    <section class="section-heading page-header">
      <div>
        <div class="eyebrow">Result Viewer</div>
        <h1>{escape(str(result['case_id']))}</h1>
        <p>{escape(result['note'])}</p>
      </div>
      <div class="hero-actions">
        <a class="button ghost" href="/samples">更多样例</a>
        <a class="button secondary" href="/run">上传新病例</a>
      </div>
    </section>

    <section class="card-grid two-column-grid">
      <article class="card info-card">
        <h2>病例信息</h2>
        <div class="info-list">
          {_render_case_info_rows(result['case_info'])}
        </div>
        <div class="metric-badge-row">{metric_badges_html}</div>
      </article>
      <article class="card status-panel">
        <h2>处理状态</h2>
        <div class="status-grid">{status_cards_html}</div>
      </article>
    </section>

    <section class="card viewer-card">
      <div class="section-heading compact">
        <div>
          <h2>3D 结果展示</h2>
          <p>直接复用已有 HTML 预览文件，通过 iframe 嵌入当前结果页。</p>
        </div>
      </div>
      {viewer_html}
    </section>

    <section class="card-grid content-split-grid">
      <article class="card slice-panel">
        <h2>2D 切片 / 叠加图</h2>
        <div class="slice-gallery slice-grid">{slice_gallery_html}</div>
      </article>
      <article class="card summary-panel">
        <h2>简短说明</h2>
        <div class="summary-copy">
          {summary_html}
        </div>
      </article>
    </section>
    """
    return render_page(title=str(result["case_id"]), body=body, active_nav="samples")


@router.get("/viewer/{result_id}")
def viewer_file(result_id: str):
    try:
        viewer_path = get_viewer_file_for_result(result_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(str(viewer_path), media_type="text/html")
