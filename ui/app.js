(function () {
  const API_BASE = window.location.pathname.replace(/\/ui\/.*$/, "/api");
  const AUTO_REFRESH_MS = 10000;

  let autoTimer = null;
  let selectedSource = null;
  let cachedConfig = null;
  let lastHealth = null;
  let discoveredModels = [];
  let pendingGgufTransformerPath = "";
  let generateDimsTouched = false;

  function byId(id) {
    return document.getElementById(id);
  }

  function qs(sel, root = document) {
    return root.querySelector(sel);
  }

  function qsa(sel, root = document) {
    return Array.from(root.querySelectorAll(sel));
  }

  function setText(id, value) {
    const node = byId(id);
    if (node) node.textContent = String(value ?? "");
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

  function escapeHtml(text) {
    return String(text ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function parseErrorPayload(payload) {
    if (payload == null) return "Unknown error";
    if (typeof payload === "string") return payload || "Unknown error";
    if (Array.isArray(payload)) return payload.length ? parseErrorPayload(payload[0]) : "Unknown error";
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
    if (!resp.ok) throw new Error(parseErrorPayload(payload));
    return payload;
  }

  const api = {
    get: (path) => request(path, { method: "GET" }),
    postJson: (path, body) => request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }),
    postForm: (path, formData) => request(path, { method: "POST", body: formData }),
    del: (path) => request(path, { method: "DELETE" }),
  };

  function setBusy(buttonId, busy, busyText) {
    const btn = byId(buttonId);
    if (!btn) return;
    if (!btn.dataset.defaultText) btn.dataset.defaultText = btn.textContent || "";
    btn.disabled = !!busy;
    btn.textContent = busy ? (busyText || "Working...") : btn.dataset.defaultText;
  }

  function setStatus(kind, text) {
    const dot = byId("status-dot");
    const label = byId("status-text");
    if (dot) {
      dot.className = "status-dot";
      if (kind) dot.classList.add(kind);
    }
    if (label) label.textContent = text || "";
  }

  function notify(kind, text) {
    setStatus(kind, text);
  }

  function switchView(nextView) {
    qsa(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === nextView));
    qsa(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${nextView}`));
  }

  function emptyNode(text) {
    const div = document.createElement("div");
    div.className = "empty-state";
    div.textContent = text;
    return div;
  }

  function setInputValue(id, value) {
    const node = byId(id);
    if (!node) return;
    if (document.activeElement === node) return;
    node.value = value == null ? "" : String(value);
  }

  const GGUF_BASE_MODEL_FALLBACK = "Wan-AI/Wan2.2-I2V-A14B-Diffusers";

  function getSelectedModelOption() {
    const select = byId("discovered-models");
    const opt = select?.options?.[select.selectedIndex] || null;
    return opt && opt.value ? opt : null;
  }

  function deriveConfigModelState() {
    const manualModel = byId("model-id-or-path")?.value?.trim() || "";
    const selectedOpt = getSelectedModelOption();
    if (selectedOpt?.dataset.modelFormat === "gguf" || selectedOpt?.dataset.modelFormat === "safetensors") {
      return {
        modelIdOrPath: selectedOpt.dataset.baseModelPath || manualModel || GGUF_BASE_MODEL_FALLBACK,
        runtimeBackend: "diffusers_gguf",
        ggufTransformerPath: selectedOpt.dataset.ggufH || pendingGgufTransformerPath || "",
      };
    }
    if (selectedOpt?.dataset.modelFormat === "diffusers") {
      return {
        modelIdOrPath: selectedOpt.dataset.modelPath || manualModel,
        runtimeBackend: "diffusers",
        ggufTransformerPath: "",
      };
    }
    const manualLooksQuantized = /\.(gguf|safetensors)$/i.test(manualModel);
    if (pendingGgufTransformerPath || manualLooksQuantized) {
      return {
        modelIdOrPath: manualLooksQuantized ? GGUF_BASE_MODEL_FALLBACK : (manualModel || GGUF_BASE_MODEL_FALLBACK),
        runtimeBackend: "diffusers_gguf",
        ggufTransformerPath: pendingGgufTransformerPath || manualModel,
      };
    }
    return {
      modelIdOrPath: manualModel,
      runtimeBackend: manualModel ? "diffusers" : "auto",
      ggufTransformerPath: "",
    };
  }

  function collectConfigForm() {
    const modelState = deriveConfigModelState();
    const payload = {
      model_id_or_path: modelState.modelIdOrPath,
      models_dir: byId("models-dir")?.value?.trim() || "",
      output_dir: byId("output-dir")?.value?.trim() || "",
      runtime_backend: modelState.runtimeBackend,
      default_width: Number(byId("default-width")?.value || 0) || null,
      default_height: Number(byId("default-height")?.value || 0) || null,
      default_frames: Number(byId("default-frames")?.value || 0) || null,
      default_fps: Number(byId("default-fps")?.value || 0) || null,
      default_steps: Number(byId("default-steps")?.value || 0) || null,
      default_guidance_scale: Number(byId("default-guidance")?.value || 0) || null,
      gguf_transformer_path: modelState.ggufTransformerPath,
    };
    return payload;
  }

  function applyConfigToForm(config) {
    if (!config) return;
    cachedConfig = config;
    setInputValue("model-id-or-path", config.model_id_or_path || "");
    setInputValue("models-dir", config.models_dir || "");
    setInputValue("output-dir", config.output_dir || "");
    pendingGgufTransformerPath = config.gguf_transformer_path || "";
    setInputValue("default-width", config.default_width || "");
    setInputValue("default-height", config.default_height || "");
    setInputValue("default-frames", config.default_frames || "");
    setInputValue("default-fps", config.default_fps || "");
    setInputValue("default-steps", config.default_steps || "");
    setInputValue("default-guidance", config.default_guidance_scale || "");
    syncDiscoveredModelSelection();
  }

  function loadGenerateDefaults() {
    const c = cachedConfig || lastHealth?.config || {};
    setInputValue("gen-width", c.default_width || 832);
    setInputValue("gen-height", c.default_height || 480);
    setInputValue("gen-frames", c.default_frames || 81);
    setInputValue("gen-fps", c.default_fps || 16);
    setInputValue("gen-steps", c.default_steps || 30);
    setInputValue("gen-guidance", c.default_guidance_scale || 5.0);
    generateDimsTouched = false;
  }

  function snapGenerateDimension(value) {
    return Math.max(256, Math.round(Number(value || 0) / 16) * 16);
  }

  function deriveGenerateDimensionsFromAspect(sourceWidth, sourceHeight) {
    return deriveGenerateDimensionsFromAspectWithArea(sourceWidth, sourceHeight, null);
  }

  function deriveGenerateDimensionsFromAspectWithArea(sourceWidth, sourceHeight, targetAreaOverride) {
    const c = cachedConfig || lastHealth?.config || {};
    const defaultWidth = Number(c.default_width || 832);
    const defaultHeight = Number(c.default_height || 480);
    const aspect = Number(sourceWidth) / Number(sourceHeight || 1);
    if (!Number.isFinite(aspect) || aspect <= 0) {
      return { width: defaultWidth, height: defaultHeight };
    }
    const targetArea = Math.max(256 * 256, Number(targetAreaOverride || 0) || defaultWidth * defaultHeight);
    const width = snapGenerateDimension(Math.sqrt(targetArea * aspect));
    const height = snapGenerateDimension(width / aspect);
    return { width, height };
  }

  function frameCountForDuration(seconds, fps) {
    const safeFps = Math.max(1, Math.round(Number(fps) || 16));
    const targetSeconds = Math.max(1, Number(seconds) || 3);
    const frameEstimate = Math.max(8, Math.round(targetSeconds * safeFps));
    const k = Math.max(2, Math.round((frameEstimate - 1) / 4));
    return Math.min(241, Math.max(9, 4 * k + 1));
  }

  function qualityPresetConfig(quality) {
    const map = {
      preview: { areaScale: 0.75, steps: 12, guidance: 4.5 },
      balanced: { areaScale: 1.0, steps: 20, guidance: 5.0 },
      high: { areaScale: 1.2, steps: 28, guidance: 5.0 },
    };
    return map[String(quality || "balanced")] || map.balanced;
  }

  function applyGeneratePreset() {
    const seconds = Number(byId("preset-seconds")?.value || 3) || 3;
    const quality = String(byId("preset-quality")?.value || "balanced");
    const preset = qualityPresetConfig(quality);
    const c = cachedConfig || lastHealth?.config || {};
    const fps = Number(c.default_fps || 16) || 16;
    const frames = frameCountForDuration(seconds, fps);
    const defaultArea = Number(c.default_width || 832) * Number(c.default_height || 480);
    let dims = {
      width: Number(c.default_width || 832),
      height: Number(c.default_height || 480),
    };

    if (selectedSource?.web_path) {
      const img = new Image();
      img.onload = () => {
        const adjusted = deriveGenerateDimensionsFromAspectWithArea(
          img.naturalWidth,
          img.naturalHeight,
          defaultArea * preset.areaScale,
        );
        setInputValue("gen-width", adjusted.width);
        setInputValue("gen-height", adjusted.height);
      };
      img.src = selectedSource.web_path;
    } else {
      dims.width = snapGenerateDimension(Math.sqrt(defaultArea * preset.areaScale * (dims.width / Math.max(1, dims.height))));
      dims.height = snapGenerateDimension(dims.width / (dims.width / Math.max(1, dims.height)));
      setInputValue("gen-width", dims.width);
      setInputValue("gen-height", dims.height);
    }

    setInputValue("gen-fps", fps);
    setInputValue("gen-frames", frames);
    setInputValue("gen-steps", preset.steps);
    setInputValue("gen-guidance", preset.guidance);
    generateDimsTouched = false;

    const actualSeconds = (frames / fps).toFixed(1);
    setText("preset-summary", `Preset applied: ${quality} • ${actualSeconds}s • ${frames} frames @ ${fps} fps • ${preset.steps} steps`);
  }

  function applySourceAspectRatioDefaults(item) {
    if (!item?.web_path || generateDimsTouched) return;
    const img = new Image();
    img.onload = () => {
      const dims = deriveGenerateDimensionsFromAspect(img.naturalWidth, img.naturalHeight);
      setInputValue("gen-width", dims.width);
      setInputValue("gen-height", dims.height);
      generateDimsTouched = false;
    };
    img.src = item.web_path;
  }

  function formatJobStage(stage) {
    const raw = String(stage || "").trim();
    if (!raw) return "";
    const labels = {
      waiting_for_gpu: "Waiting for GPU",
      preparing_gpu: "Preparing GPU",
      loading_pipeline: "Loading pipeline",
      loading_transformer: "Loading transformer",
      assembling_pipeline: "Assembling pipeline",
      pipeline_loaded: "Pipeline loaded",
      fallback_offload: "Switching to CPU offload",
      generating: "Generating frames",
      writing_outputs: "Writing outputs",
      completed: "Completed",
      canceled: "Canceled",
      cancel_requested: "Cancel requested",
      failed: "Failed",
    };
    return labels[raw] || raw.replaceAll("_", " ");
  }

  function renderReadiness(health) {
    const ready = health?.ready || {};
    const install = health?.installation || {};
    const runtime = health?.runtime || {};
    const diffusers = install.diffusers || {};
    const compatibility = install.compatibility || {};

    const badges = [
      {
        label: "Model Source",
        ok: !!install.model_source_configured,
        value: install.model_source_configured ? `${install.model_source_kind || "configured"}` : "Not configured",
      },
      {
        label: "Diffusers Wan",
        ok: !!install.diffusers_ready,
        value: install.diffusers_ready ? `Ready (${diffusers.version || "unknown"})` : (diffusers.error || "Missing"),
      },
      {
        label: "Output Dir",
        ok: !!install.output_dir_writable,
        value: install.output_dir_writable ? "Writable" : (install.output_dir_error || "Not writable"),
      },
      {
        label: "Runtime",
        ok: !!ready.runtime_ready,
        value: ready.runtime_ready ? `${runtime.profile?.device || "?"} / ${runtime.profile?.dtype || "?"}` : (runtime.error || "Unavailable"),
      },
      {
        label: "Compatibility",
        ok: compatibility.compatible !== false,
        value: compatibility.compatible === false
          ? "Incompatible with current setup"
          : (compatibility.runtime_plan?.selected_backend === "diffusers_gguf"
            ? "Compatible (GGUF backend)"
            : "Compatible"),
      },
      {
        label: "Engine Ready",
        ok: !!ready.engine_ready,
        value: ready.engine_ready ? "Ready to queue jobs" : "Setup incomplete",
      },
    ];

    const wrap = byId("readiness-badges");
    if (wrap) {
      wrap.innerHTML = "";
      badges.forEach((b) => {
        const el = document.createElement("article");
        el.className = `badge ${b.ok ? "ok" : "warn"}`;
        el.innerHTML = `
          <div class="badge-label">${escapeHtml(b.label)}</div>
          <div class="badge-value">${escapeHtml(String(b.value || ""))}</div>
        `;
        wrap.appendChild(el);
      });
    }

    const summary = byId("readiness-summary");
    if (summary) {
      const model = install.model_source || "unset";
      const outputDir = install.output_dir || "(default)";
      if (compatibility.compatible === false) {
        summary.textContent = `Model: ${model}. Output dir: ${outputDir}. Incompatible with current setup. ${compatibility.reason || "DuckMotion cannot safely run this backend on this machine."}`;
      } else {
        summary.textContent = `Model: ${model}. Output dir: ${outputDir}. ${ready.engine_ready ? "DuckMotion local runtime is ready." : "Complete setup items before generating."}`;
      }
      summary.classList.remove("muted");
    }

    const missingList = byId("missing-list");
    if (missingList) {
      missingList.innerHTML = "";
      const missing = Array.isArray(install.missing) ? install.missing : [];
      const compatReason = compatibility.compatible === false ? String(compatibility.reason || "Incompatible with current setup.") : "";
      if (!missing.length && !compatReason) {
        const li = document.createElement("li");
        li.className = "empty";
        li.textContent = "No setup blockers detected.";
        missingList.appendChild(li);
      } else {
        if (compatReason) {
          const li = document.createElement("li");
          li.textContent = compatReason;
          missingList.appendChild(li);
        }
        missing.forEach((row) => {
          const li = document.createElement("li");
          li.textContent = row;
          missingList.appendChild(li);
        });
      }
    }

    renderRuntimeSummary(health);
    renderStatusPill(health);
  }

  function renderStatusPill(health) {
    const ready = health?.ready || {};
    const compatibility = health?.installation?.compatibility || {};
    if (compatibility.compatible === false) {
      setStatus("error", "Incompatible Setup");
      return;
    }
    if (ready.engine_ready) {
      setStatus("ok", "DuckMotion Ready");
      return;
    }
    if (ready.plugin_enabled || health?.installation?.model_source_configured) {
      setStatus("warn", "Setup Needed");
      return;
    }
    setStatus("error", "Not Configured");
  }

  function renderRuntimeSummary(health) {
    const wrap = byId("runtime-summary");
    if (!wrap) return;
    wrap.innerHTML = "";

    const runtime = health?.runtime || {};
    const profile = runtime?.profile || {};
    const install = health?.installation || {};
    const diffusers = install?.diffusers || {};
    const runtimePlan = install?.compatibility?.runtime_plan || install?.runtime_plan || {};

    const selectedBackend = runtimePlan.selected_backend || install.runtime_backend || "auto";
    const ggufCandidate = runtimePlan.gguf_candidate || install.runtime_plan?.gguf_candidate || null;
    const ggufLabel = ggufCandidate
      ? `${ggufCandidate.label || ggufCandidate.family || "GGUF"} (${ggufCandidate.complete ? "H+L pair" : "partial"})`
      : (runtimePlan.gguf_transformer_path ? runtimePlan.gguf_transformer_path.split(/[\\/]/).pop() : null);

    const rows = [
      { label: "Device", value: profile.device || "unknown" },
      { label: "DType", value: profile.dtype || "unknown" },
      { label: "GPU", value: profile.cuda_device_name || "Not detected" },
      { label: "VRAM", value: profile.total_vram_gb ? `${profile.total_vram_gb} GB` : "--" },
      { label: "Diffusers", value: diffusers.version || "unknown" },
      {
        label: "Backend",
        value: selectedBackend === "diffusers_gguf"
          ? `GGUF${ggufLabel ? " \u2014 " + ggufLabel : ""}`
          : selectedBackend,
      },
      ...(runtimePlan.gguf_available && runtimePlan.available_backends?.includes("diffusers_gguf")
        ? [{ label: "GGUF Status", value: selectedBackend === "diffusers_gguf" ? "Active" : "Available" }]
        : []),
    ];
    rows.forEach((row) => wrap.appendChild(buildStatCard(row.label, row.value)));
  }

  function renderEngineStatus(payload) {
    const wrap = byId("engine-summary");
    if (!wrap) return;
    wrap.innerHTML = "";

    const queue = payload?.queue || {};
    const lease = payload?.gpu_lease?.lease || {};
    const rows = [
      { label: "Pipeline", value: payload?.pipeline_loaded ? "Loaded" : "Not loaded" },
      { label: "Queued", value: queue.queued ?? 0 },
      { label: "Running", value: queue.running ?? 0 },
      { label: "Cancel Requests", value: queue.cancel_requested ?? 0 },
      { label: "Output Dir", value: payload?.output_dir || "--" },
      { label: "GPU Lease", value: payload?.gpu_lease?.held ? `${lease.owner || "unknown"} (${lease.label || "active"})` : "Available" },
    ];
    rows.forEach((row) => wrap.appendChild(buildStatCard(row.label, row.value)));
  }

  function buildStatCard(label, value) {
    const card = document.createElement("article");
    card.className = "stat-card";
    card.innerHTML = `
      <div class="stat-label">${escapeHtml(label)}</div>
      <div class="stat-value">${escapeHtml(String(value ?? "--"))}</div>
    `;
    return card;
  }

  function setGenerateFeedback(kind, message) {
    const node = byId("generate-feedback");
    if (!node) return;
    node.className = "callout";
    if (kind === "error") node.classList.add("error-callout");
    if (kind === "warn") node.classList.add("warn-callout");
    if (!kind || kind === "muted") node.classList.add("muted");
    node.textContent = message || "";
  }

  function sourceLabel(source) {
    return source === "checkpoint_wan" ? "checkpoint/wan"
      : source === "checkpoint_wan_plugin" ? "checkpoint/wan (plugin root)"
      : source === "checkpoint_wan_cwd" ? "checkpoint/wan (cwd)"
      : source === "configured_models_dir" ? "configured dir"
      : source === "hf_cache" ? "HF cache"
      : source;
  }

  function renderDiscoveredModels(payload) {
    const select = byId("discovered-models");
    const hint = byId("discovered-models-hint");
    if (!select) return;

    const items = Array.isArray(payload?.items) ? payload.items : [];
    const ggufCandidates = Array.isArray(payload?.gguf_candidates) ? payload.gguf_candidates : [];
    discoveredModels = items;

    select.innerHTML = "";

    const totalCount = items.length + ggufCandidates.length;
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = totalCount
      ? `Choose model (${totalCount} found)...`
      : "No local Wan models found (manual entry still supported)";
    select.appendChild(placeholder);

    // --- GGUF candidates (shown first so they are the preferred selection) ---
    if (ggufCandidates.length) {
      const grp = document.createElement("optgroup");
      grp.label = "GGUF (quantized)";
      ggufCandidates.forEach((candidate) => {
        const hPath = candidate.paths?.H || candidate.paths?.single || "";
        if (!hPath) return;
        const option = document.createElement("option");
        option.value = `gguf:${hPath}`;
        const quant = candidate.quant ? ` ${candidate.quant}` : "";
        const pair = candidate.complete ? " [H+L pair]" : " [partial]";
        option.textContent = `${candidate.label || candidate.family || "GGUF"}${quant}${pair}`;
        option.dataset.ggufH = hPath;
        option.dataset.ggufL = candidate.paths?.L || "";
        option.dataset.baseModelPath = candidate.base_model_path || "";
        option.dataset.modelFormat = "gguf";
        grp.appendChild(option);
      });
      select.appendChild(grp);
    }

    // --- Diffusers model directories ---
    if (items.length) {
      const grp = document.createElement("optgroup");
      grp.label = "Diffusers (full model)";
      items.forEach((item) => {
        const option = document.createElement("option");
        option.value = String(item.path || "");
        option.textContent = `${item.label || item.path} [${sourceLabel(item.source)}]`;
        option.dataset.modelPath = String(item.path || "");
        option.dataset.modelFormat = "diffusers";
        if (item.repo_id) option.dataset.repoId = String(item.repo_id);
        grp.appendChild(option);
      });
      select.appendChild(grp);
    }

    const modelInput = byId("model-id-or-path");
    const current = String(modelInput?.value || "").trim();
    if (!current) {
      // Prefer a GGUF candidate if available, otherwise first diffusers model.
      const preferredGguf = ggufCandidates[0];
      const preferredDiffusers = items.find((item) => String(item.source || "") === "checkpoint_wan") || items[0];
      if (preferredGguf && modelInput) {
        const hPath = preferredGguf.paths?.H || preferredGguf.paths?.single || "";
        if (hPath) {
          select.value = `gguf:${hPath}`;
          modelInput.value = preferredGguf.base_model_path || preferredDiffusers?.path || GGUF_BASE_MODEL_FALLBACK;
          pendingGgufTransformerPath = hPath;
        }
      } else if (preferredDiffusers?.path && modelInput) {
        modelInput.value = String(preferredDiffusers.path);
        pendingGgufTransformerPath = "";
      } else if (payload?.default_model_id && modelInput) {
        modelInput.value = String(payload.default_model_id);
        pendingGgufTransformerPath = "";
      }
    }
    syncDiscoveredModelSelection();

    // Wire change handler for the combined dropdown.
    select.onchange = null;
    select.addEventListener("change", async () => {
      const opt = select.options[select.selectedIndex];
      if (!opt || !opt.value) return;
      const modelInput = byId("model-id-or-path");
      if (opt.dataset.modelFormat === "gguf" || opt.dataset.modelFormat === "safetensors") {
        const hPath = opt.dataset.ggufH || "";
        const basePath = opt.dataset.baseModelPath || GGUF_BASE_MODEL_FALLBACK;
        pendingGgufTransformerPath = hPath;
        if (modelInput) modelInput.value = basePath || modelInput.value;
      } else {
        pendingGgufTransformerPath = "";
        if (modelInput) modelInput.value = opt.dataset.modelPath || opt.value;
      }
      try {
        await saveConfig();
        await Promise.all([loadHealth(), loadEngineStatus()]);
      } catch (err) {
        notify("error", `Failed to auto-save model selection: ${err.message || String(err)}`);
      }
    });

    if (hint) {
      const roots = Array.isArray(payload?.scan_roots) ? payload.scan_roots : [];
      const existingCount = roots.filter((r) => r && r.exists).length;
      const errors = Array.isArray(payload?.scan_errors) ? payload.scan_errors : [];
      const ggufInfo = ggufCandidates.length ? ` ${ggufCandidates.length} GGUF/quantized model(s) found.` : "";
      const defaultInfo = payload?.default_model_id ? ` Default remote: ${payload.default_model_id}.` : "";
      hint.innerHTML =
        `Discovered ${items.length} diffusers model(s) across ${existingCount}/${roots.length} search roots.${ggufInfo}${defaultInfo}` +
        `${errors.length ? ` ${errors.length} scan error(s).` : ""}`;
    }
  }

  function syncDiscoveredModelSelection() {
    const select = byId("discovered-models");
    const input = byId("model-id-or-path");
    if (!select || !input) return;
    if (pendingGgufTransformerPath) {
      const ggufValue = `gguf:${pendingGgufTransformerPath}`;
      const ggufOption = Array.from(select.options || []).find((opt) => opt.value === ggufValue);
      if (ggufOption) {
        select.value = ggufValue;
        return;
      }
    }
    const current = String(input.value || "").trim();
    if (!current) {
      select.value = "";
      return;
    }
    const exact = discoveredModels.find((item) => String(item.path || "") === current);
    select.value = exact ? current : "";
  }

  function updateSelectedSource(item) {
    selectedSource = item || null;
    const box = byId("selected-source");
    if (!box) return;
    if (!selectedSource) {
      box.className = "callout muted";
      box.textContent = "No source image selected.";
      return;
    }
    box.className = "callout";
    box.innerHTML = `Selected: <strong>${escapeHtml(selectedSource.name || selectedSource.path || "image")}</strong><br><span class="muted">${escapeHtml(selectedSource.web_path || selectedSource.path || "")}</span>`;
    applySourceAspectRatioDefaults(selectedSource);
  }

  function renderStaging(items) {
    const wrap = byId("staging-list");
    if (!wrap) return;
    wrap.innerHTML = "";
    if (!Array.isArray(items) || !items.length) {
      wrap.appendChild(emptyNode("No staged images yet."));
      return;
    }
    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "media-item";
      const src = item.web_path || "";
      card.innerHTML = `
        <img class="media-thumb" alt="${escapeHtml(item.name || "staged image")}" src="${escapeHtml(src)}" />
        <div class="media-meta">
          <div class="media-name">${escapeHtml(item.name || "")}</div>
          <div class="media-sub">${formatBytes(item.size_bytes)} • ${formatTime(item.mtime)}</div>
          <div class="media-actions">
            <button class="btn btn-primary btn-sm" type="button" data-action="select">Use</button>
            <button class="btn btn-secondary btn-sm" type="button" data-action="delete">Delete</button>
          </div>
        </div>
      `;
      card.addEventListener("click", async (e) => {
        const btn = e.target.closest("button");
        if (!btn) return;
        const action = btn.dataset.action;
        if (action === "select") {
          updateSelectedSource(item);
          return;
        }
        if (action === "delete") {
          try {
            await api.del(`/staging/${encodeURIComponent(item.name)}`);
            if (selectedSource && selectedSource.name === item.name) updateSelectedSource(null);
            await loadStaging();
          } catch (err) {
            notify("error", err.message || String(err));
          }
        }
      });
      wrap.appendChild(card);
    });
  }

  function renderRecent(items) {
    const wrap = byId("recent-list");
    if (!wrap) return;
    wrap.innerHTML = "";
    if (!Array.isArray(items) || !items.length) {
      wrap.appendChild(emptyNode("No recent WebbDuck images found."));
      return;
    }
    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "media-item";
      card.innerHTML = `
        <img class="media-thumb" alt="${escapeHtml(item.name || "recent image")}" src="${escapeHtml(item.web_path || "")}" />
        <div class="media-meta">
          <div class="media-name">${escapeHtml(item.name || "")}</div>
          <div class="media-sub">${escapeHtml(item.run || "run")} • ${formatTime(item.mtime)}</div>
          <div class="media-actions">
            <button class="btn btn-secondary btn-sm" type="button" data-action="copy">Copy to Staging</button>
          </div>
        </div>
      `;
      card.addEventListener("click", async (e) => {
        const btn = e.target.closest("button[data-action='copy']");
        if (!btn) return;
        const fd = new FormData();
        fd.append("path", item.web_path || item.path || "");
        try {
          await api.postForm("/staging/from-webbduck", fd);
          notify("ok", "Copied image to staging");
          await loadStaging();
        } catch (err) {
          notify("error", err.message || String(err));
        }
      });
      wrap.appendChild(card);
    });
  }

  function jobStatusClass(status) {
    const value = String(status || "unknown").toLowerCase();
    if (["completed"].includes(value)) return "completed";
    if (["failed", "error"].includes(value)) return "bad";
    if (["running"].includes(value)) return "running";
    if (["queued", "cancel_requested"].includes(value)) return "queued";
    if (["canceled"].includes(value)) return "warn";
    return "";
  }

  function renderJobs(jobs) {
    const wrap = byId("jobs-list");
    if (!wrap) return;
    wrap.innerHTML = "";
    if (!Array.isArray(jobs) || !jobs.length) {
      wrap.appendChild(emptyNode("No DuckMotion jobs yet."));
      return;
    }

    jobs.forEach((job) => {
      const item = document.createElement("div");
      item.className = "job-item";
      const status = String(job.status || "unknown");
      const progress = job.progress || {};
      const progressDetail = String(progress.detail || "").trim();
      const outputs = Array.isArray(job.outputs) ? job.outputs : [];
      const params = job.params || {};
      const input = job.input || {};
      const warnings = Array.isArray(job.warnings) ? job.warnings : [];
      item.innerHTML = `
        <div class="job-head">
          <div class="job-id">${escapeHtml(job.job_id || "job")}</div>
          <div class="job-status ${escapeHtml(jobStatusClass(status))}">${escapeHtml(status)}</div>
        </div>
        <div class="job-meta">
          ${escapeHtml(formatTime(job.created_at))} • ${escapeHtml(formatJobStage(progress.stage || ""))}${progress.percent != null ? ` • ${escapeHtml(String(progress.percent))}%` : ""}${progressDetail ? ` • ${escapeHtml(progressDetail)}` : ""}
        </div>
        <div class="job-meta">Input: ${escapeHtml(input.image_name || input.image_path || "")}</div>
        <div class="job-meta">${escapeHtml((params.prompt || "").slice(0, 220) || "(no prompt)")}</div>
        ${job.error ? `<div class="callout error-callout">${escapeHtml(job.error)}</div>` : ""}
        ${warnings.length ? `<div class="callout warn-callout">${warnings.map((w) => escapeHtml(w)).join("<br>")}</div>` : ""}
        <div class="row">
          ${(status === "queued" || status === "running" || status === "cancel_requested") ? '<button class="btn btn-secondary btn-sm" type="button" data-action="cancel">Cancel</button>' : ""}
        </div>
        <div class="job-outputs"></div>
      `;

      const outputsWrap = qs(".job-outputs", item);
      if (outputsWrap) {
        if (!outputs.length) {
          outputsWrap.appendChild(emptyNode("No outputs yet."));
        } else {
          outputs.forEach((out) => {
            outputsWrap.appendChild(renderOutputCard(out));
          });
        }
      }

      item.addEventListener("click", async (e) => {
        const btn = e.target.closest("button[data-action='cancel']");
        if (!btn) return;
        try {
          await api.postJson("/engine/cancel", { job_id: job.job_id });
          notify("warn", "Cancel requested");
          await loadJobs();
        } catch (err) {
          notify("error", err.message || String(err));
        }
      });

      wrap.appendChild(item);
    });
  }

  function renderOutputCard(out) {
    const card = document.createElement("div");
    card.className = "output-card";
    const posterUrl = out?.poster?.url || out?.poster?.web_path || "";
    const videoUrl = out?.video?.url || out?.video?.web_path || "";
    const meta = out?.meta || {};
    const frameCount = meta.frame_count ?? out?.meta?.frame_count;

    if (videoUrl) {
      const video = document.createElement("video");
      video.src = videoUrl;
      video.controls = true;
      video.preload = "metadata";
      if (posterUrl) video.poster = posterUrl;
      card.appendChild(video);
    } else if (posterUrl) {
      const img = document.createElement("img");
      img.src = posterUrl;
      img.alt = "DuckMotion poster";
      card.appendChild(img);
    }

    const title = document.createElement("div");
    title.className = "media-name";
    title.textContent = out.run_id || out?.video?.name || "Output";
    card.appendChild(title);

    const sub = document.createElement("div");
    sub.className = "media-sub";
    sub.textContent = `${frameCount || (meta?.frame_count ?? "?")} frames • ${formatTime(out.mtime || meta.created_at)}`;
    card.appendChild(sub);

    if (videoUrl) {
      const a = document.createElement("a");
      a.href = videoUrl;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = out?.video?.name || "Open video";
      card.appendChild(a);
    }

    return card;
  }

  function renderGallery(items) {
    const wrap = byId("gallery-list");
    if (!wrap) return;
    wrap.innerHTML = "";
    if (!Array.isArray(items) || !items.length) {
      wrap.appendChild(emptyNode("No generated videos yet."));
      return;
    }
    items.forEach((item) => wrap.appendChild(renderOutputCard(item)));
  }

  async function loadHealth() {
    const health = await api.get("/health");
    lastHealth = health;
    renderReadiness(health);
    applyConfigToForm(health.config || {});
    if (!byId("gen-width")?.value) loadGenerateDefaults();
    return health;
  }

  async function loadConfig() {
    const data = await api.get("/config");
    applyConfigToForm(data.config || {});
    return data;
  }

  async function saveConfig() {
    setBusy("save-config", true, "Saving...");
    try {
      const payload = collectConfigForm();
      const data = await api.postJson("/config", payload);
      applyConfigToForm(data.config || {});
      notify("ok", "DuckMotion config saved");
      await Promise.all([loadHealth(), loadEngineStatus(), loadDiscoveredModels()]);
    } catch (err) {
      notify("error", err.message || String(err));
    } finally {
      setBusy("save-config", false);
    }
  }

  async function loadEngineStatus() {
    const data = await api.get("/engine/status");
    renderEngineStatus(data);
    return data;
  }

  async function loadStaging() {
    const data = await api.get("/staging?limit=100");
    renderStaging(data.items || []);
    if (selectedSource) {
      const found = (data.items || []).find((x) => x.name === selectedSource.name);
      if (!found) updateSelectedSource(null);
    }
    return data;
  }

  async function loadRecent() {
    const data = await api.get("/webbduck/recent-images?limit=48");
    renderRecent(data.items || []);
    return data;
  }

  async function loadJobs() {
    const data = await api.get("/engine/jobs?limit=60");
    renderJobs(data.jobs || []);
    return data;
  }

  async function loadGallery() {
    const data = await api.get("/gallery?limit=120");
    renderGallery(data.items || []);
    return data;
  }

  async function loadDiscoveredModels() {
    const data = await api.get("/models/discover");
    renderDiscoveredModels(data || {});
    return data;
  }

  async function uploadStagingFile(file) {
    if (!file) throw new Error("No file selected.");
    const fd = new FormData();
    fd.append("image", file);
    return api.postForm("/staging/upload", fd);
  }

  async function copyWebbDuckImageToStaging(path) {
    const fd = new FormData();
    fd.append("path", path);
    return api.postForm("/staging/from-webbduck", fd);
  }

  async function submitGenerate() {
    if (!selectedSource?.web_path && !selectedSource?.path) {
      throw new Error("Select a source image from staging first.");
    }
    const prompt = (byId("prompt")?.value || "").trim();
    if (!prompt) throw new Error("Prompt is required.");

    const payload = {
      image_path: selectedSource.web_path || selectedSource.path,
      prompt,
      negative_prompt: (byId("negative-prompt")?.value || "").trim(),
      width: Number(byId("gen-width")?.value || 0) || null,
      height: Number(byId("gen-height")?.value || 0) || null,
      num_frames: Number(byId("gen-frames")?.value || 0) || null,
      fps: Number(byId("gen-fps")?.value || 0) || null,
      num_inference_steps: Number(byId("gen-steps")?.value || 0) || null,
      guidance_scale: Number(byId("gen-guidance")?.value || 0) || null,
      seed: (byId("gen-seed")?.value || "").trim() ? Number(byId("gen-seed").value) : null,
    };

    return api.postJson("/engine/generate", payload);
  }

  async function refreshAll() {
    try {
      await Promise.all([
        loadHealth(),
        loadEngineStatus(),
        loadStaging(),
        loadRecent(),
        loadJobs(),
        loadGallery(),
      ]);
    } catch (err) {
      notify("error", err.message || String(err));
    }
  }

  async function handleWebbDuckHandoffMessage(payload) {
    const imageSrc = String(payload?.image?.src || "").trim();
    if (!imageSrc) return;
    if (!imageSrc.startsWith("/outputs/")) return;

    try {
      const copied = await copyWebbDuckImageToStaging(imageSrc);
      if (copied?.item) {
        updateSelectedSource(copied.item);
      }

      const meta = payload?.meta || {};
      if (typeof meta.prompt === "string" && meta.prompt.trim() && !byId("prompt")?.value?.trim()) {
        byId("prompt").value = meta.prompt;
      }
      if (typeof meta.negative_prompt === "string" && !byId("negative-prompt")?.value?.trim()) {
        byId("negative-prompt").value = meta.negative_prompt;
      }
      if (Number.isFinite(Number(meta.width)) && !byId("gen-width")?.value) {
        byId("gen-width").value = String(meta.width);
      }
      if (Number.isFinite(Number(meta.height)) && !byId("gen-height")?.value) {
        byId("gen-height").value = String(meta.height);
      }
      if (meta.seed != null && !byId("gen-seed")?.value) {
        byId("gen-seed").value = String(meta.seed);
      }

      switchView("create");
      await Promise.all([loadStaging(), loadRecent()]);
      notify("ok", "Image received from WebbDuck");
    } catch (err) {
      notify("error", err.message || String(err));
    }
  }

  function bindEvents() {
    qsa(".tab").forEach((tab) => {
      tab.addEventListener("click", () => switchView(tab.dataset.view));
    });

    byId("refresh-all")?.addEventListener("click", refreshAll);
    byId("check-health")?.addEventListener("click", async () => {
      setBusy("check-health", true, "Checking...");
      try {
        await Promise.all([loadHealth(), loadEngineStatus()]);
      } catch (err) {
        notify("error", err.message || String(err));
      } finally {
        setBusy("check-health", false);
      }
    });
    byId("save-config")?.addEventListener("click", saveConfig);
    byId("model-id-or-path")?.addEventListener("input", syncDiscoveredModelSelection);
    byId("refresh-models")?.addEventListener("click", async () => {
      setBusy("refresh-models", true, "Scanning...");
      try {
        await loadDiscoveredModels();
        notify("ok", "Model discovery refreshed");
      } catch (err) {
        notify("error", err.message || String(err));
      } finally {
        setBusy("refresh-models", false);
      }
    });
    byId("unload-engine")?.addEventListener("click", async () => {
      setBusy("unload-engine", true, "Unloading...");
      try {
        await api.postJson("/engine/unload", {});
        notify("ok", "Model unloaded");
        await loadEngineStatus();
      } catch (err) {
        notify("error", err.message || String(err));
      } finally {
        setBusy("unload-engine", false);
      }
    });

    byId("fill-defaults")?.addEventListener("click", loadGenerateDefaults);
    byId("apply-generate-preset")?.addEventListener("click", applyGeneratePreset);
    byId("gen-width")?.addEventListener("input", () => {
      generateDimsTouched = true;
    });
    byId("gen-height")?.addEventListener("input", () => {
      generateDimsTouched = true;
    });

    byId("upload-form")?.addEventListener("submit", async (e) => {
      e.preventDefault();
      const input = byId("upload-file");
      const file = input?.files?.[0];
      setBusy("upload-btn", true, "Uploading...");
      try {
        const data = await uploadStagingFile(file);
        if (data?.item) updateSelectedSource(data.item);
        if (input) input.value = "";
        notify("ok", "Image uploaded to staging");
        await loadStaging();
      } catch (err) {
        notify("error", err.message || String(err));
      } finally {
        setBusy("upload-btn", false);
      }
    });

    byId("refresh-staging")?.addEventListener("click", loadStaging);
    byId("refresh-recent")?.addEventListener("click", loadRecent);

    byId("submit-generate")?.addEventListener("click", async () => {
      setBusy("submit-generate", true, "Queueing...");
      try {
        const data = await submitGenerate();
        const job = data?.job || {};
        const detail = `Queued ${job.job_id || "job"} • ${job.params?.width || "--"}x${job.params?.height || "--"} • ${job.params?.num_frames || "--"} frames`;
        setGenerateFeedback("ok", detail);
        notify("ok", `Queued ${job.job_id || "job"}`);
        switchView("jobs");
        await loadJobs();
      } catch (err) {
        setGenerateFeedback("error", err.message || String(err));
        notify("error", err.message || String(err));
      } finally {
        setBusy("submit-generate", false);
      }
    });

    byId("refresh-jobs")?.addEventListener("click", loadJobs);
    byId("clear-jobs")?.addEventListener("click", async () => {
      try {
        await api.postJson("/jobs/clear", {});
        notify("ok", "Job list cleared");
        await loadJobs();
      } catch (err) {
        notify("error", err.message || String(err));
      }
    });

    byId("refresh-gallery")?.addEventListener("click", loadGallery);

    window.addEventListener("message", async (event) => {
      if (event.origin !== window.location.origin) return;
      const payload = event.data;
      if (!payload || typeof payload !== "object") return;
      if (payload.type !== "webbduck.duckmotion.handoff") return;
      await handleWebbDuckHandoffMessage(payload);
    });
  }

  function startAutoRefresh() {
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = setInterval(async () => {
      try {
        await Promise.all([loadJobs(), loadGallery(), loadEngineStatus()]);
      } catch (_err) {
        // Avoid noisy polling errors in the UI; manual refresh will surface details.
      }
    }, AUTO_REFRESH_MS);
  }

  async function init() {
    bindEvents();
    setStatus("warn", "Loading...");
    try {
      await Promise.all([loadConfig(), refreshAll(), loadDiscoveredModels()]);
      startAutoRefresh();
    } catch (err) {
      setStatus("error", err.message || String(err));
    }
  }

  window.addEventListener("beforeunload", () => {
    if (autoTimer) clearInterval(autoTimer);
  });

  init();
})();
