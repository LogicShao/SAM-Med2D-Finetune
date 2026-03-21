from __future__ import annotations

from html import escape

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from web_demo.services.cases import list_sample_cases
from web_demo.ui.common import render_page


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home_page() -> HTMLResponse:
    sample_cases = list_sample_cases()
    body = f"""
    <section class="hero card hero-card">
      <div class="eyebrow">Minimal Demo</div>
      <h1>上传/选择病例，串行跑通自动分割、后处理和 3D 结果查看</h1>
      <p class="lead">页面只保留系统主链路，不再展示训练曲线、复杂对比页、大段指标看板和无关操作按钮。</p>
      <div class="hero-actions">
        <a class="button primary" href="/samples">选择样例病例</a>
        <a class="button secondary" href="/run">上传病例并运行</a>
      </div>
      <div class="hero-meta">当前已发现 {len(sample_cases)} 个稳定样例，可直接查看结果，不重新计算。</div>
    </section>

    <section class="card-grid two-column-grid">
      <article class="card">
        <h2>主链路</h2>
        <ol class="step-list">
          <li>选择已有病例结果，或上传单病例目录/文件。</li>
          <li>调用现有整病例推理脚本完成自动分割。</li>
          <li>调用现有后处理和 3D 可视化逻辑生成结果目录。</li>
          <li>在结果页查看病例信息、状态、3D HTML 和 2D 叠加切片。</li>
        </ol>
      </article>
      <article class="card">
        <h2>当前约束</h2>
        <ul class="bullet-list">
          <li>只保证单病例串行 demo，不做队列、数据库和并发管理。</li>
          <li>优先保证“选择样例病例 - 查看结果”稳定可演示。</li>
          <li>上传链路复用现有脚本，不额外改写算法流程。</li>
        </ul>
      </article>
    </section>
    """
    return render_page(title="首页", body=body, active_nav="home")


@router.get("/samples", response_class=HTMLResponse)
def sample_cases_page() -> HTMLResponse:
    sample_cases = list_sample_cases()
    if not sample_cases:
        cards_html = """
        <div class="empty-state">
          <h2>未找到可直接回放的样例结果</h2>
          <p>请先确认 outputs 中已有带 case_meta.json 和 3D HTML 预览的病例目录。</p>
        </div>
        """
    else:
        cards_html = "".join(
            f"""
            <a class="case-card" href="/results/{escape(case.result_id)}">
              <div class="case-card-top">
                <span class="pill">{escape(case.source_label)}</span>
                <span class="muted">{escape(case.source_tag)}</span>
              </div>
              <h3>{escape(case.case_id)}</h3>
              <p>{escape(case.summary)}</p>
              <span class="case-link">进入结果页</span>
            </a>
            """
            for case in sample_cases
        )

    body = f"""
    <section class="section-heading page-header">
      <div>
        <div class="eyebrow">Sample Cases</div>
        <h1>选择样例病例</h1>
        <p>样例页面只读复用已有 outputs 结果，选中后直接进入结果页，不重新计算。</p>
      </div>
      <a class="button ghost" href="/">返回首页</a>
    </section>
    <section class="case-grid sample-grid">
      {cards_html}
    </section>
    """
    return render_page(title="样例病例", body=body, active_nav="samples")
