(function () {
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

  window.WebDemo = {
    initRunForm,
    initRunWaitPage,
  };
})();
