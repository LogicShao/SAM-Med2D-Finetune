(function () {
  function preserveModeLinks() {
    const url = new URL(window.location.href);
    const mode = url.searchParams.get("mode");
    if (!mode) {
      return;
    }

    document.querySelectorAll("a[href]").forEach((link) => {
      const href = link.getAttribute("href");
      if (!href || href.startsWith("#") || href.startsWith("javascript:")) {
        return;
      }
      if (!href.startsWith("/")) {
        return;
      }
      if (href.includes("mode=")) {
        return;
      }

      const targetUrl = new URL(href, window.location.origin);
      targetUrl.searchParams.set("mode", mode);
      link.setAttribute("href", `${targetUrl.pathname}${targetUrl.search}${targetUrl.hash}`);
    });
  }

  function initRunForm() {
    const form = document.getElementById("run-form");
    const submitButton = document.getElementById("run-submit");
    if (!form || !submitButton) {
      return;
    }

    form.addEventListener(
      "submit",
      function () {
        submitButton.disabled = true;
        submitButton.textContent = "正在接收病例...";
        submitButton.classList.add("is-disabled");
      },
      { once: true }
    );
  }

  function initRunWaitPage() {
    const root = document.getElementById("run-wait-root");
    const initialStateNode = document.getElementById("run-initial-state");
    if (!root || !initialStateNode) {
      return;
    }

    const statusUrl = root.dataset.statusUrl;
    const pollIntervalMs = 1500;
    let redirectStarted = false;

    const chip = document.getElementById("job-status-chip");
    const currentStage = document.getElementById("job-current-stage");
    const message = document.getElementById("job-message");
    const stageGrid = document.getElementById("job-stage-grid");
    const logPanel = document.getElementById("job-log-panel");
    const resultLink = document.getElementById("job-result-link");
    const redirectNote = document.getElementById("job-redirect-note");

    let initialState = {};
    try {
      initialState = JSON.parse(initialStateNode.textContent || "{}");
    } catch (error) {
      initialState = {};
    }

    const statusTextMap = {
      running: "处理中",
      success: "已完成",
      failed: "失败",
      pending: "等待中",
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function renderStages(stages) {
      stageGrid.innerHTML = (stages || [])
        .map(
          (stage) => `
            <article class="job-stage-card is-${escapeHtml(stage.state)}">
              <div class="job-stage-top">
                <span class="job-stage-label">${escapeHtml(stage.label)}</span>
                <span class="job-stage-state">${escapeHtml(statusTextMap[stage.state] || stage.state)}</span>
              </div>
              <p>${escapeHtml(stage.message || "等待执行")}</p>
            </article>
          `
        )
        .join("");
    }

    function renderLogs(lines) {
      const text = (lines || []).length ? lines.join("\n") : "等待日志输出...";
      logPanel.textContent = text;
      logPanel.scrollTop = logPanel.scrollHeight;
    }

    function renderState(data) {
      chip.textContent = statusTextMap[data.status] || data.status || "处理中";
      chip.className = `job-status-chip is-${data.status || "running"}`;
      currentStage.textContent = data.current_stage_label || data.current_stage || "处理中";
      message.textContent = data.message || "后台正在执行，请稍候。";
      renderStages(data.stages);
      renderLogs(data.log_tail);

      if (data.result_url) {
        resultLink.hidden = false;
        resultLink.href = data.result_url;
      }

      if (data.status === "success" && data.result_url && !redirectStarted) {
        redirectStarted = true;
        redirectNote.hidden = false;
        window.setTimeout(function () {
          window.location.href = data.result_url;
        }, 1200);
      }
    }

    function poll() {
      fetch(statusUrl, { cache: "no-store" })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          return response.json();
        })
        .then((data) => {
          renderState(data);
          if (data.status === "running" || data.status === "pending") {
            window.setTimeout(poll, pollIntervalMs);
          }
        })
        .catch((error) => {
          message.textContent = `状态轮询失败：${error.message}`;
          window.setTimeout(poll, pollIntervalMs);
        });
    }

    renderState(initialState);
    if (initialState.status === "running" || initialState.status === "pending") {
      window.setTimeout(poll, pollIntervalMs);
    }
  }

  function initResultViewerControls() {
    const panel = document.getElementById("viewer-control-panel");
    const iframe = document.getElementById("result-viewer-frame");
    if (!panel || !iframe) {
      return;
    }

    const statusNode = document.getElementById("viewer-control-status");
    const buttons = Array.from(panel.querySelectorAll(".viewer-toggle"));
    const state = { WT: true, TC: true, ET: true };
    let directControlAvailable = null;
    let fallbackActive = false;

    function activeMasks() {
      return ["WT", "TC", "ET"].filter((mask) => state[mask]);
    }

    function updateButtonCopy() {
      buttons.forEach((button) => {
        const mask = button.dataset.mask;
        const isActive = Boolean(state[mask]);
        button.classList.toggle("is-active", isActive);
        button.classList.toggle("is-inactive", !isActive);
        button.setAttribute("aria-pressed", isActive ? "true" : "false");
        button.textContent = `${mask}\uff1a${isActive ? "\u663e\u793a" : "\u9690\u85cf"}`;
      });
    }

    function selectionToMaskParam() {
      const masks = activeMasks();
      if (masks.length === 3) {
        return "all";
      }
      return masks.join(",");
    }

    function buildFallbackUrl() {
      const baseUrl = new URL(panel.dataset.defaultSrc || iframe.src, window.location.origin);
      baseUrl.searchParams.set("mask", selectionToMaskParam());
      return `${baseUrl.pathname}${baseUrl.search}${baseUrl.hash}`;
    }

    function detectTraceMask(traceName) {
      const text = String(traceName || "").toUpperCase();
      if (text.includes("WT")) {
        return "WT";
      }
      if (text.includes("TC")) {
        return "TC";
      }
      if (text.includes("ET")) {
        return "ET";
      }
      return null;
    }

    function tryApplyDirectControl() {
      try {
        const frameWindow = iframe.contentWindow;
        const frameDocument = iframe.contentDocument;
        const plotly = frameWindow && frameWindow.Plotly;
        const graph = frameDocument && frameDocument.querySelector(".plotly-graph-div");
        const traces = graph && (graph.data || graph._fullData);
        if (!plotly || !graph || !Array.isArray(traces) || !traces.length) {
          return false;
        }

        const indices = [];
        const visibilities = [];
        traces.forEach((trace, index) => {
          const mask = detectTraceMask(trace.name);
          if (!mask) {
            return;
          }
          indices.push(index);
          visibilities.push(Boolean(state[mask]));
        });
        if (!indices.length) {
          return false;
        }

        plotly.restyle(graph, { visible: visibilities }, indices);
        return true;
      } catch (error) {
        return false;
      }
    }

    function applyViewerState() {
      updateButtonCopy();

      if (directControlAvailable !== false) {
        const applied = tryApplyDirectControl();
        if (applied) {
          directControlAvailable = true;
          fallbackActive = false;
          if (statusNode) {
            statusNode.textContent = "\u5f53\u524d\u4e3a\u4ea4\u4e92\u663e\u9690\u63a7\u5236\u3002";
          }
          return;
        }
      }

      directControlAvailable = false;
      const nextUrl = buildFallbackUrl();
      if (iframe.getAttribute("src") !== nextUrl) {
        fallbackActive = true;
        iframe.setAttribute("src", nextUrl);
      }
      if (statusNode) {
        statusNode.textContent =
          "\u5f53\u524d viewer \u4e0d\u652f\u6301\u76f4\u63a5\u663e\u9690\uff0c\u5df2\u5207\u6362\u4e3a\u5bf9\u5e94\u5206\u533a\u89c6\u56fe\u3002";
      }
    }

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const mask = button.dataset.mask;
        const activeCount = activeMasks().length;
        if (state[mask] && activeCount === 1) {
          return;
        }
        state[mask] = !state[mask];
        applyViewerState();
      });
    });

    iframe.addEventListener("load", () => {
      if (!fallbackActive) {
        directControlAvailable = tryApplyDirectControl();
        if (directControlAvailable && statusNode) {
          statusNode.textContent = "\u5f53\u524d\u4e3a\u4ea4\u4e92\u663e\u9690\u63a7\u5236\u3002";
        }
      }
      fallbackActive = false;
    });

    updateButtonCopy();
    window.setTimeout(applyViewerState, 120);
  }

  window.WebDemo = {
    initRunForm,
    initRunWaitPage,
    initResultViewerControls,
  };

  window.addEventListener("DOMContentLoaded", function () {
    preserveModeLinks();
    initResultViewerControls();
  });
})();
