(function () {
  const API_BASE = window.location.pathname.replace(/\/ui\/.*$/, "/api");
  const AUTO_REFRESH_MS = 20000;

  let autoRefreshTimer = null;
  let lastHealth = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    const node = byId(id);
    if (!node) return;
    node.textContent = String(value ?? "");
  }

  function pretty(value) {
    try {
      return JSON.stringify(value, null, 2);
    } catch (_e) {
      return String(value);
    }
  }

  function formatBytes(num) {
    const n = Number(num);
    if (!Number.isFinite(n) || n < 0) return "--";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
    return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }

  function formatTime(epochSec) {
    const n = Number(epochSec);
    if (!Number.isFinite(n) || n <= 0) return "unknown";
    return new Date(n * 1000).toLocaleString();
  }

  function showJson(id, value) {
    setText(id, pretty(value));
  }

  function setBusy(buttonId, busy, busyText) {
    const btn = byId(buttonId);
    if (!btn) return;
    if (!btn.dataset.defaultText) btn.dataset.defaultText = btn.textContent || "";
    btn.disabled = !!busy;
    btn.textContent = busy ? (busyText || "Working...") : btn.dataset.defaultText;
  }

  function parseErrorPayload(payload) {
    if (payload == null) return "Unknown error";
    if (typeof payload === "string") {
      const trimmed = payload.trim();
      if (!trimmed) return "Unknown error";
      try {
        return parseErrorPayload(JSON.parse(trimmed));
      } catch (_e) {
        return trimmed;
      }
    }
    if (Array.isArray(payload)) {
      return payload.length ? parseErrorPayload(payload[0]) : "Unknown error";
    }
    if (typeof payload === "object") {
      if (typeof payload.detail === "string") return payload.detail;
      if (payload.detail && typeof payload.detail === "object") {
        if (typeof payload.detail.error === "string") return payload.detail.error;
        if (typeof payload.detail.message === "string") return payload.detail.message;
      }
      if (typeof payload.error === "string") return payload.error;
      if (typeof payload.message === "string") return payload.message;
      return pretty(payload);
    }
    return String(payload);
  }

  async function request(path, options = {}) {
    const resp = await fetch(`${API_BASE}${path}`, options);
    const ct = resp.headers.get("content-type") || "";
    const payload = ct.includes("application/json") ? await resp.json() : await resp.text();
    if (!resp.ok) {
      throw new Error(parseErrorPayload(payload));
    }
    return payload;
  }

  function get(path) {
    return request(path, { method: "GET" });
  }

  function postJson(path, body) {
    return request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  }

  function postForm(path, formData) {
    return request(path, { method: "POST", body: formData });
  }

  function del(path) {
    return request(path, { method: "DELETE" });
  }

  function switchView(nextView) {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.view === nextView);
    });
    document.querySelectorAll(".view").forEach((view) => {
      view.classList.toggle("active", view.id === `view-${nextView}`);
    });
  }

  function setGlobalStatus(kind, text) {
    const dot = byId("status-dot");
    const label = byId("status-text");
    if (dot) {
      dot.className = "status-dot";
      if (kind) dot.classList.add(kind);
    }
    if (label) label.textContent = text || "";
  }

  function notify(kind, message) {
    setGlobalStatus(kind, message);
  }

  function renderReadiness(health) {
    const ready = health?.ready || {};
    const install = health?.installation || {};
    const comfy = health?.comfy || {};

    const badges = [
      {
        label: "Plugin Enabled",
        ok: !!ready.plugin_enabled,
        warn: !ready.plugin_enabled,
        value: ready.plugin_enabled ? "GGUF pair found" : "Need high+low I2V GGUFs",
      },
      {
        label: "Companion Files",
        ok: !!ready.generation_prereqs,
        warn: !!install.gguf_pair_found && !ready.generation_prereqs,
        value: ready.generation_prereqs ? "Text encoder + VAE found" : "Missing text encoder / VAE",
      },
      {
        label: "ComfyUI",
        ok: !!ready.comfy_connected,
        warn: comfy?.configured && !ready.comfy_connected,
        value: ready.comfy_connected ? "Connected" : (comfy?.configured ? "Configured but unreachable" : "Not configured"),
      },
      {
        label: "Bridge Ready",
        ok: !!ready.bridge_ready,
        warn: !ready.bridge_ready,
        value: ready.bridge_ready ? "Ready for raw prompt bridge" : "Complete setup checks first",
      },
    ];

    const badgeWrap = byId("readiness-badges");
    if (badgeWrap) {
      badgeWrap.innerHTML = "";
      badges.forEach((b) => {
        const article = document.createElement("article");
        article.className = `badge ${b.ok ? "ok" : (b.warn ? "warn" : "bad")}`;
        article.innerHTML = `
          <div class="badge-label">${escapeHtml(b.label)}</div>
          <div class="badge-value">${escapeHtml(b.value)}</div>
        `;
        badgeWrap.appendChild(article);
      });
    }

    const summary = byId("readiness-summary");
    if (summary) {
      const missingCount = Array.isArray(install.missing) ? install.missing.length : 0;
      const pair = install.best_pair?.quant ? `Best detected pair: ${install.best_pair.quant}` : "No I2V GGUF pair detected yet";
      const comfyLine = comfy?.reachable ? `ComfyUI connected: ${comfy.base_url}` : (comfy?.error || "ComfyUI not configured");
      summary.textContent = `${pair}. ${comfyLine}. Missing checks: ${missingCount}.`;
      summary.classList.toggle("muted", false);
    }

    renderPairs(install.pairs || []);
    renderMissing(install);
  }

  function renderPairs(pairs) {
    const list = byId("pair-list");
    if (!list) return;
    list.innerHTML = "";
    if (!Array.isArray(pairs) || !pairs.length) {
      list.appendChild(emptyLi("No Wan2.2 I2V GGUF high/low pair detected."));
      return;
    }
    pairs.forEach((pair) => {
      const li = document.createElement("li");
      li.innerHTML = `
        <strong>${escapeHtml(pair.quant || "Unknown")}</strong> · ${escapeHtml(String(pair.total_size_gb || 0))} GB total<br/>
        <span class="muted">${escapeHtml(pair.high?.name || "")}</span><br/>
        <span class="muted">${escapeHtml(pair.low?.name || "")}</span>
      `;
      list.appendChild(li);
    });
  }

  function renderMissing(install) {
    const list = byId("missing-list");
    if (!list) return;
    list.innerHTML = "";
    const missing = Array.isArray(install?.missing) ? install.missing : [];
    const scanErrors = Array.isArray(install?.scan_errors) ? install.scan_errors : [];

    if (!missing.length && !scanErrors.length) {
      list.appendChild(emptyLi("No missing prerequisites detected for Phase 1 checks."));
      return;
    }

    missing.forEach((row) => {
      const li = document.createElement("li");
      li.textContent = row;
      list.appendChild(li);
    });
    scanErrors.forEach((row) => {
      const li = document.createElement("li");
      li.textContent = `Scan error: ${row}`;
      list.appendChild(li);
    });
  }

  function emptyLi(text) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = text;
    return li;
  }

  function renderStatusFromHealth(health) {
    const ready = health?.ready || {};
    if (ready.bridge_ready) {
      setGlobalStatus("ok", "Bridge Ready");
      return;
    }
    if (ready.plugin_enabled || health?.comfy?.configured) {
      setGlobalStatus("warn", "Setup Needed");
      return;
    }
    setGlobalStatus("error", "Not Configured");
  }

  async function loadHealth() {
    const health = await get("/health");
    lastHealth = health;

    const config = health?.config || {};
    if (byId("comfy-url") && document.activeElement !== byId("comfy-url")) byId("comfy-url").value = config.comfy_url || "";
    if (byId("models-dir") && document.activeElement !== byId("models-dir")) byId("models-dir").value = config.models_dir || "";
    if (byId("comfy-models-dir") && document.activeElement !== byId("comfy-models-dir")) byId("comfy-models-dir").value = config.comfy_models_dir || "";

    renderReadiness(health);
    renderStatusFromHealth(health);
    return health;
  }

  async function saveConfig() {
    setBusy("save-config", true, "Saving...");
    try {
      await postJson("/config", {
        comfy_url: byId("comfy-url")?.value || "",
        models_dir: byId("models-dir")?.value || "",
        comfy_models_dir: byId("comfy-models-dir")?.value || "",
      });
      await loadHealth();
    } finally {
      setBusy("save-config", false);
    }
  }

  function renderMediaList(containerId, items, options = {}) {
    const root = byId(containerId);
    if (!root) return;
    root.innerHTML = "";
    if (!Array.isArray(items) || !items.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = options.emptyText || "No items.";
      root.appendChild(empty);
      return;
    }

    items.forEach((item) => {
      const row = document.createElement("article");
      row.className = "media-item";
      const thumbSrc = item.web_path || item.url || "";
      const name = item.name || item.filename || "item";
      const sub = item.run ? `Run ${item.run}` : (item.subfolder || item.path || "");

      row.innerHTML = `
        <img class="media-thumb" src="${escapeHtml(thumbSrc)}" alt="${escapeHtml(name)}" loading="lazy" />
        <div class="media-meta">
          <div class="media-name">${escapeHtml(name)}</div>
          <div class="media-sub">${escapeHtml(sub || "")}</div>
          <div class="media-sub">${item.size_bytes != null ? escapeHtml(formatBytes(item.size_bytes)) : ""}</div>
          <div class="media-actions"></div>
        </div>
      `;
      const actions = row.querySelector(".media-actions");
      if (actions) {
        if (options.onStage) {
          const stageBtn = document.createElement("button");
          stageBtn.className = "btn btn-secondary btn-sm";
          stageBtn.type = "button";
          stageBtn.textContent = "Stage";
          stageBtn.addEventListener("click", () => options.onStage(item, stageBtn));
          actions.appendChild(stageBtn);
        }
        if (options.onDelete) {
          const delBtn = document.createElement("button");
          delBtn.className = "btn btn-secondary btn-sm";
          delBtn.type = "button";
          delBtn.textContent = "Remove";
          delBtn.addEventListener("click", () => options.onDelete(item, delBtn));
          actions.appendChild(delBtn);
        }
        if (thumbSrc) {
          const open = document.createElement("a");
          open.className = "btn btn-secondary btn-sm";
          open.href = thumbSrc;
          open.target = "_blank";
          open.rel = "noopener noreferrer";
          open.textContent = "Open";
          actions.appendChild(open);
        }
      }
      root.appendChild(row);
    });
  }

  async function loadStaging() {
    const payload = await get("/staging");
    renderMediaList("staging-list", payload?.items || [], {
      emptyText: "No staged images yet.",
      onDelete: async (item, btn) => {
        if (!item?.name) return;
        btn.disabled = true;
        try {
          await del(`/staging/${encodeURIComponent(item.name)}`);
          await loadStaging();
        } catch (err) {
          notify("error", err.message);
        } finally {
          btn.disabled = false;
        }
      },
    });
  }

  async function loadRecentWebbduckImages() {
    const payload = await get("/webbduck/recent-images?limit=24");
    renderMediaList("recent-list", payload?.items || [], {
      emptyText: "No recent WebbDuck images found.",
      onStage: async (item, btn) => {
        if (!item?.web_path) return;
        btn.disabled = true;
        const form = new FormData();
        form.append("path", item.web_path);
        try {
          await postForm("/staging/from-webbduck", form);
          await loadStaging();
          switchView("setup");
        } catch (err) {
          notify("error", err.message);
        } finally {
          btn.disabled = false;
        }
      },
    });
  }

  async function uploadToStaging(event) {
    event.preventDefault();
    const input = byId("upload-file");
    const file = input?.files?.[0];
    if (!file) {
      notify("warn", "Choose an image first.");
      return;
    }

    setBusy("upload-btn", true, "Uploading...");
    try {
      const form = new FormData();
      form.append("image", file);
      await postForm("/staging/upload", form);
      if (input) input.value = "";
      await loadStaging();
    } finally {
      setBusy("upload-btn", false);
    }
  }

  async function submitPromptJson() {
    const raw = byId("prompt-json")?.value || "";
    let payload;
    try {
      payload = JSON.parse(raw);
    } catch (err) {
      showJson("bridge-result", { error: `Invalid JSON: ${err.message}` });
      return;
    }
    setBusy("submit-prompt-json", true, "Submitting...");
    try {
      const result = await postJson("/comfy/prompt", payload);
      showJson("bridge-result", result);
      await loadJobs(true);
      switchView("jobs");
    } catch (err) {
      showJson("bridge-result", { error: err.message });
    } finally {
      setBusy("submit-prompt-json", false);
    }
  }

  function loadExampleJson() {
    const sample = {
      prompt: {
        "1": {
          inputs: {
            text: "Replace this with your Wan prompt",
            clip: ["2", 0]
          },
          class_type: "CLIPTextEncode"
        }
      },
      client_id: "duckmotion-example"
    };
    byId("prompt-json").value = pretty(sample);
  }

  async function loadComfyQueue() {
    setBusy("refresh-comfy-queue", true, "Loading...");
    try {
      const data = await get("/comfy/queue");
      showJson("comfy-queue-json", data);
    } catch (err) {
      showJson("comfy-queue-json", { error: err.message });
    } finally {
      setBusy("refresh-comfy-queue", false);
    }
  }

  async function loadComfyStats() {
    setBusy("refresh-comfy-stats", true, "Loading...");
    try {
      const data = await get("/comfy/system_stats");
      showJson("comfy-stats-json", data);
    } catch (err) {
      showJson("comfy-stats-json", { error: err.message });
    } finally {
      setBusy("refresh-comfy-stats", false);
    }
  }

  function renderJobs(jobs) {
    const root = byId("jobs-list");
    if (!root) return;
    root.innerHTML = "";
    if (!Array.isArray(jobs) || !jobs.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No tracked bridge jobs yet.";
      root.appendChild(empty);
      return;
    }

    jobs.forEach((job) => {
      const article = document.createElement("article");
      article.className = "job-item";
      const status = String(job?.status || "unknown").toLowerCase();
      const outputs = Array.isArray(job?.history_outputs) ? job.history_outputs : [];

      article.innerHTML = `
        <div class="job-head">
          <div class="job-id">${escapeHtml(job.prompt_id || "unknown")}</div>
          <div class="job-status ${escapeHtml(status)}">${escapeHtml(status)}</div>
        </div>
        <div class="job-meta">
          Submitted: ${escapeHtml(formatTime(job.submitted_at))} · Phase: ${escapeHtml(job.phase || "bridge")}
          ${job.history_error ? ` · History error: ${escapeHtml(job.history_error)}` : ""}
        </div>
        <div class="row">
          <button class="btn btn-secondary btn-sm" type="button" data-refresh-job="${escapeHtml(job.prompt_id || "")}">Refresh Job</button>
          <button class="btn btn-secondary btn-sm" type="button" data-load-history="${escapeHtml(job.prompt_id || "")}">Load Raw History</button>
        </div>
        <pre class="json-box muted job-history-box" id="job-history-${escapeHtml(job.prompt_id || "")}">History summary: ${escapeHtml(pretty(job.history_raw || {}))}</pre>
        <div class="job-outputs"></div>
      `;

      const outWrap = article.querySelector(".job-outputs");
      if (outWrap) {
        if (!outputs.length) {
          const empty = document.createElement("div");
          empty.className = "empty-state";
          empty.textContent = "No outputs discovered yet.";
          outWrap.appendChild(empty);
        } else {
          outputs.forEach((output) => {
            const card = document.createElement("div");
            card.className = "output-card";
            const url = String(output?.url || "");
            const filename = String(output?.filename || "output");
            const mediaKind = String(output?.media_kind || "file");
            if (mediaKind === "video") {
              const video = document.createElement("video");
              video.src = url;
              video.controls = true;
              video.preload = "metadata";
              card.appendChild(video);
            } else if (mediaKind === "image") {
              const img = document.createElement("img");
              img.src = url;
              img.loading = "lazy";
              img.alt = filename;
              card.appendChild(img);
            }
            const link = document.createElement("a");
            link.href = url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = filename;
            card.appendChild(link);
            outWrap.appendChild(card);
          });
        }
      }

      root.appendChild(article);
    });

    root.querySelectorAll("[data-refresh-job]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const promptId = btn.getAttribute("data-refresh-job");
        if (!promptId) return;
        btn.disabled = true;
        try {
          await get(`/jobs/${encodeURIComponent(promptId)}?refresh=true`);
          await loadJobs(false);
        } catch (err) {
          notify("error", err.message);
        } finally {
          btn.disabled = false;
        }
      });
    });

    root.querySelectorAll("[data-load-history]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const promptId = btn.getAttribute("data-load-history");
        if (!promptId) return;
        const box = byId(`job-history-${promptId}`);
        btn.disabled = true;
        try {
          const data = await get(`/comfy/history/${encodeURIComponent(promptId)}`);
          if (box) box.textContent = pretty(data);
        } catch (err) {
          if (box) box.textContent = pretty({ error: err.message });
        } finally {
          btn.disabled = false;
        }
      });
    });
  }

  async function loadJobs(refresh = true) {
    setBusy("refresh-jobs", true, "Loading...");
    try {
      const payload = await get(`/jobs?limit=40&refresh=${refresh ? "true" : "false"}`);
      renderJobs(payload?.jobs || []);
    } catch (err) {
      renderJobs([]);
      const root = byId("jobs-list");
      if (root) {
        const note = document.createElement("div");
        note.className = "empty-state";
        note.textContent = `Failed to load jobs: ${err.message}`;
        root.prepend(note);
      }
    } finally {
      setBusy("refresh-jobs", false);
    }
  }

  async function clearJobs() {
    setBusy("clear-jobs", true, "Clearing...");
    try {
      await postJson("/jobs/clear", {});
      await loadJobs(false);
    } finally {
      setBusy("clear-jobs", false);
    }
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  async function refreshAll() {
    try {
      await Promise.all([
        loadHealth(),
        loadStaging().catch(() => {}),
        loadRecentWebbduckImages().catch(() => {}),
        loadJobs(true).catch(() => {}),
      ]);
    } catch (err) {
      setGlobalStatus("error", err.message || "Refresh failed");
    }
  }

  function scheduleAutoRefresh() {
    if (autoRefreshTimer) clearTimeout(autoRefreshTimer);
    autoRefreshTimer = window.setTimeout(async () => {
      try {
        await loadHealth();
      } catch (_err) {
        // keep the UI usable even if periodic refresh fails
      } finally {
        scheduleAutoRefresh();
      }
    }, AUTO_REFRESH_MS);
  }

  function bindEvents() {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => switchView(tab.dataset.view || "setup"));
    });

    byId("refresh-all")?.addEventListener("click", refreshAll);
    byId("save-config")?.addEventListener("click", () => saveConfig().catch(showTopError));
    byId("check-health")?.addEventListener("click", () => loadHealth().catch(showTopError));

    byId("upload-form")?.addEventListener("submit", (e) => uploadToStaging(e).catch(showTopError));
    byId("refresh-staging")?.addEventListener("click", () => loadStaging().catch(showTopError));
    byId("refresh-recent")?.addEventListener("click", () => loadRecentWebbduckImages().catch(showTopError));

    byId("submit-prompt-json")?.addEventListener("click", () => submitPromptJson().catch(showTopError));
    byId("load-example-json")?.addEventListener("click", loadExampleJson);
    byId("refresh-comfy-queue")?.addEventListener("click", () => loadComfyQueue().catch(showTopError));
    byId("refresh-comfy-stats")?.addEventListener("click", () => loadComfyStats().catch(showTopError));

    byId("refresh-jobs")?.addEventListener("click", () => loadJobs(true).catch(showTopError));
    byId("clear-jobs")?.addEventListener("click", () => clearJobs().catch(showTopError));
  }

  function showTopError(err) {
    const msg = err?.message || "Error";
    setGlobalStatus("error", msg);
    console.error(err);
  }

  async function init() {
    bindEvents();
    loadExampleJson();
    await refreshAll();
    scheduleAutoRefresh();
  }

  init().catch((err) => {
    console.error(err);
    setGlobalStatus("error", err?.message || "Init failed");
  });
})();
