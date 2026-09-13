(function () {
  const API_BASE = window.location.pathname.replace(/\/ui\/.*$/, "/api");
  const AUTO_REFRESH_MS = 10000;

  let cachedConfig = null;
  let discoveredModels = [];
  let activeModel = null;
  let selectedSource = null;
  let lastAppliedModelKey = "";
  let autoTimer = null;
  let generateDimsTouched = false;

  function byId(id) {
    return document.getElementById(id);
  }

  function qsa(selector, root = document) {
    return Array.from(root.querySelectorAll(selector));
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function parseError(payload) {
    if (payload == null) return "Unknown error";
    if (typeof payload === "string") return payload || "Unknown error";
    if (typeof payload === "object") {
      if (typeof payload.detail === "string") return payload.detail;
      if (payload.detail && typeof payload.detail === "object") {
        if (payload.detail.message) return String(payload.detail.message);
        if (payload.detail.error) return String(payload.detail.error);
      }
      if (payload.message) return String(payload.message);
      if (payload.error) return String(payload.error);
      try { return JSON.stringify(payload); } catch (_e) { return String(payload); }
    }
    return String(payload);
  }

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const type = response.headers.get("content-type") || "";
    const payload = type.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) throw new Error(parseError(payload));
    return payload;
  }

  const api = {
    get: (path) => request(path, { method: "GET" }),
    postJson: (path, body) => request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }),
    postForm: (path, body) => request(path, { method: "POST", body }),
    del: (path) => request(path, { method: "DELETE" }),
  };

  function setStatus(kind, text) {
    const dot = byId("status-dot");
    const label = byId("status-text");
    if (dot) {
      dot.className = "status-dot";
      if (kind) dot.classList.add(kind);
    }
    if (label) label.textContent = text || "";
  }

  function setBusy(id, busy, text = "Working…") {
    const button = byId(id);
    if (!button) return;
    if (!button.dataset.label) button.dataset.label = button.textContent || "";
    button.disabled = !!busy;
    button.textContent = busy ? text : button.dataset.label;
  }

  function setInput(id, value) {
    const input = byId(id);
    if (!input || document.activeElement === input) return;
    input.value = value == null ? "" : String(value);
  }

  function setHidden(id, hidden) {
    byId(id)?.classList.toggle("hidden", !!hidden);
  }

  function setText(id, value) {
    const node = byId(id);
    if (node) node.textContent = String(value ?? "");
  }

  function switchView(name) {
    qsa(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
    qsa(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  }

  function modelPersistedSource(item) {
    if (!item) return "";
    if (item.location === "hf_cache" && item.repo_id) return String(item.repo_id);
    return String(item.source || item.repo_id || item.name || "");
  }

  function modelKey(item) {
    return item ? `${item.name || ""}|${modelPersistedSource(item)}` : "";
  }

  function findConfiguredModel(source) {
    const target = String(source || "").trim();
    if (!target) return null;
    return discoveredModels.find((item) => {
      const values = [item.source, item.repo_id, item.name, modelPersistedSource(item)]
        .filter(Boolean)
        .map(String);
      return values.includes(target);
    }) || null;
  }

  function stepsLocked(constraints = {}) {
    if (constraints.steps_locked != null) return !!constraints.steps_locked;
    return !!constraints.sampling_schedule_locked;
  }

  function guidanceLocked(constraints = {}) {
    return constraints.guidance_locked != null ? !!constraints.guidance_locked : false;
  }

  function capabilityChips(model) {
    const caps = model?.capabilities || {};
    const chips = [];
    if (caps.text_to_video) chips.push("Text → Video");
    if (caps.image_to_video) chips.push(caps.source_image_required ? "Image → Video · image required" : "Image → Video · image optional");
    if (caps.video_to_video) chips.push("Video → Video");
    if (caps.audio_output) chips.push("Synchronized audio");
    if (caps.negative_prompt) chips.push("Negative prompt");
    if (!chips.length) chips.push("No runnable generation workflow");
    return chips;
  }

  function constraintText(model) {
    const constraints = model?.constraints || {};
    const bits = [];
    const multiple = Number(constraints.dimension_multiple || 0);
    if (multiple > 1) bits.push(`dimensions: ${multiple}px grid`);
    if (constraints.frame_count_modulo != null && constraints.frame_count_remainder != null) {
      bits.push(`frames: ${constraints.frame_count_modulo}k+${constraints.frame_count_remainder}`);
    }
    if (Number(constraints.generation_stages || 0) > 1) bits.push(`${constraints.generation_stages}-stage generation`);
    if (stepsLocked(constraints)) bits.push("model-defined step schedule");
    if (guidanceLocked(constraints)) bits.push("model-defined guidance");
    return bits.join(" • ");
  }

  function defaultText(model) {
    const d = model?.defaults || {};
    const parts = [];
    if (d.width && d.height) parts.push(`${d.width}×${d.height}`);
    if (d.num_frames) parts.push(`${d.num_frames} frames`);
    if (d.fps) parts.push(`${d.fps} fps`);
    if (d.num_inference_steps && !stepsLocked(model?.constraints || {})) parts.push(`${d.num_inference_steps} steps`);
    if (d.guidance_scale != null && !guidanceLocked(model?.constraints || {})) parts.push(`guidance ${d.guidance_scale}`);
    return parts.join(" • ");
  }

  function profileMarkup(model) {
    if (!model) return "Select a model to see its capabilities.";
    const chips = capabilityChips(model)
      .map((chip) => `<span class="model-profile-chip${model.supported === false ? " unavailable" : ""}">${escapeHtml(chip)}</span>`)
      .join("");
    const constraints = constraintText(model) || "No special constraints reported";
    const defaults = defaultText(model) || "Model defaults available at generation time";
    return `
      <div class="model-profile-title">
        <span>${escapeHtml(model.name || "Selected model")}</span>
        <span class="model-profile-chip${model.supported === false ? " unavailable" : ""}">${model.supported === false ? "Runtime unavailable" : "Runnable"}</span>
      </div>
      <div class="model-profile-chips">${chips}</div>
      <div class="model-profile-meta">
        <div><strong>Defaults</strong><span>${escapeHtml(defaults)}</span></div>
        <div><strong>Constraints</strong><span>${escapeHtml(constraints)}</span></div>
      </div>`;
  }

  function renderModelProfile() {
    const setup = byId("selected-model-summary");
    const create = byId("create-model-profile");
    const markup = profileMarkup(activeModel);
    if (setup) setup.innerHTML = markup;
    if (create) create.innerHTML = markup;
  }

  function renderModelSelect() {
    const select = byId("discovered-models");
    if (!select) return;
    const configured = String(cachedConfig?.model_id_or_path || "").trim();
    select.innerHTML = '<option value="">Select a model…</option>';
    discoveredModels.forEach((item, index) => {
      if (!item?.name) return;
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = item.supported === false
        ? `${item.name} — runtime unavailable`
        : item.name;
      option.disabled = item.supported === false;
      const persisted = modelPersistedSource(item);
      if ([item.source, item.repo_id, item.name, persisted].filter(Boolean).map(String).includes(configured)) {
        option.selected = true;
      }
      select.appendChild(option);
    });

    const selectedIndex = Number(select.value);
    if (select.value !== "" && Number.isInteger(selectedIndex) && selectedIndex >= 0 && discoveredModels[selectedIndex]) {
      activeModel = discoveredModels[selectedIndex];
    } else {
      activeModel = findConfiguredModel(configured);
    }
    renderModelProfile();
    applyCapabilityUI({ applyDefaults: modelKey(activeModel) !== lastAppliedModelKey });
  }

  function currentTargetSeconds() {
    const explicit = Number(byId("preset-seconds")?.value || 0);
    if (Number.isFinite(explicit) && explicit > 0) return explicit;
    const frames = Number(byId("gen-frames")?.value || activeModel?.defaults?.num_frames || 1);
    const fps = Number(byId("gen-fps")?.value || activeModel?.defaults?.fps || 1);
    return Math.max(1, (Math.max(1, frames) - 1) / Math.max(1, fps));
  }

  function applyModelDefaults(force = false) {
    if (!activeModel) return;
    const key = modelKey(activeModel);
    if (!force && key === lastAppliedModelKey) return;
    const d = activeModel.defaults || {};
    const values = {
      "gen-width": d.width,
      "gen-height": d.height,
      "gen-frames": d.num_frames,
      "gen-fps": d.fps,
      "gen-steps": d.num_inference_steps,
      "gen-guidance": d.guidance_scale,
    };
    Object.entries(values).forEach(([id, value]) => {
      if (value !== undefined && value !== null) setInput(id, value);
    });
    const fps = Number(d.fps || 0);
    const frames = Number(d.num_frames || 0);
    if (fps > 0 && frames > 1) {
      const seconds = Math.max(1, (frames - 1) / fps);
      setInput("preset-seconds", Number(seconds.toFixed(2)));
    }
    setInput("preset-quality", "balanced");
    generateDimsTouched = false;
    lastAppliedModelKey = key;
    updatePresetSummary();
  }

  function clearSelectedSource() {
    selectedSource = null;
    const box = byId("selected-source");
    if (box) {
      box.className = "callout muted";
      box.textContent = "No source image selected.";
    }
  }

  function renderSelectedSource() {
    const box = byId("selected-source");
    if (!box) return;
    if (!selectedSource) {
      box.className = "callout muted";
      box.textContent = "No source image selected.";
      return;
    }
    box.className = "callout";
    box.innerHTML = `Selected: <strong>${escapeHtml(selectedSource.name || "image")}</strong>`;
  }

  function applyCapabilityUI({ applyDefaults = false } = {}) {
    renderModelProfile();
    const caps = activeModel?.capabilities || {};
    const constraints = activeModel?.constraints || {};
    const supportsImage = !!caps.image_to_video;
    setHidden("source-panel", !supportsImage);
    setHidden("recent-panel", !supportsImage);
    setHidden("negative-prompt-group", !caps.negative_prompt);

    const stabilityModes = Array.isArray(constraints.i2v_stability_modes)
      ? constraints.i2v_stability_modes.map(String)
      : [];
    const supportsStability = supportsImage && stabilityModes.length > 0;
    setHidden("i2v-stability-group", !supportsStability);
    setHidden("i2v-stability-hint", !supportsStability);
    const stabilitySelect = byId("i2v-stability");
    if (stabilitySelect && supportsStability) {
      qsa("option", stabilitySelect).forEach((option) => {
        option.hidden = !stabilityModes.includes(option.value);
        option.disabled = !stabilityModes.includes(option.value);
      });
      const requestedDefault = String(constraints.i2v_stability_default || stabilityModes[0] || "model");
      if (!stabilityModes.includes(stabilitySelect.value)) {
        stabilitySelect.value = stabilityModes.includes(requestedDefault) ? requestedDefault : stabilityModes[0];
      }
    }

    const sourcePanel = byId("source-panel");
    sourcePanel?.classList.toggle("source-required", !!caps.source_image_required);
    const requirement = byId("source-requirement");
    if (requirement) {
      requirement.textContent = caps.source_image_required
        ? "A source image is required for this model."
        : "A source image is optional. Leave it empty for text-to-video.";
    }
    if (!supportsImage) clearSelectedSource();

    const lockedSteps = stepsLocked(constraints);
    const lockedGuidance = guidanceLocked(constraints);
    setHidden("gen-steps-group", lockedSteps);
    setHidden("gen-guidance-group", lockedGuidance);
    const note = byId("sampling-note");
    if (note) {
      const messages = [];
      if (lockedSteps) messages.push("This checkpoint defines its own step/sigma schedule.");
      if (!lockedGuidance && activeModel?.defaults?.guidance_scale != null) {
        messages.push("Guidance remains adjustable.");
      } else if (lockedGuidance) {
        messages.push("Guidance is also model-defined.");
      }
      note.classList.toggle("hidden", messages.length === 0);
      note.textContent = messages.join(" ");
    }

    const multiple = Math.max(1, Number(constraints.dimension_multiple || 1));
    for (const id of ["gen-width", "gen-height"]) {
      const input = byId(id);
      if (input) input.step = String(multiple);
    }
    const frameModulo = Math.max(1, Number(constraints.frame_count_modulo || 1));
    const frames = byId("gen-frames");
    if (frames) frames.step = String(frameModulo);

    const summary = byId("constraint-summary");
    if (summary) summary.textContent = constraintText(activeModel);

    if (applyDefaults) applyModelDefaults(true);
    updatePresetSummary();
    updateGenerateAvailability();
  }

  function updateGenerateAvailability() {
    const button = byId("submit-generate");
    if (!button) return;
    const caps = activeModel?.capabilities || {};
    let reason = "";
    if (!activeModel) reason = "Select a model in Setup.";
    else if (activeModel.supported === false) reason = "The selected model runtime is unavailable.";
    else if (caps.source_image_required && !selectedSource) reason = "Select a source image for this model.";
    else if (!caps.text_to_video && !selectedSource) reason = "This model requires supported source media.";
    button.disabled = !!reason;
    button.title = reason;
    if (reason) {
      const feedback = byId("generate-feedback");
      if (feedback && !feedback.dataset.jobFeedback) {
        feedback.className = "callout muted";
        feedback.textContent = reason;
      }
    }
  }

  function snap(value, multiple) {
    const n = Math.max(multiple, Number(value || multiple));
    return Math.max(multiple, Math.floor(n / multiple) * multiple);
  }

  function normalizeFrames(value, constraints) {
    let frames = Math.max(1, Math.floor(Number(value || 1)));
    if (constraints.frame_count_modulo != null && constraints.frame_count_remainder != null) {
      const mod = Math.max(1, Number(constraints.frame_count_modulo));
      const rem = Number(constraints.frame_count_remainder);
      frames = Math.max(rem + mod, Math.floor((Math.max(frames, rem) - rem) / mod) * mod + rem);
    }
    return frames;
  }

  function frameCountForDuration(seconds, fps, constraints) {
    const safeFps = Math.max(1, Math.round(Number(fps) || 1));
    const safeSeconds = Math.max(0.1, Number(seconds) || 1);
    const target = Math.max(2, Math.round(safeSeconds * safeFps) + 1);
    if (constraints.frame_count_modulo != null && constraints.frame_count_remainder != null) {
      const mod = Math.max(1, Number(constraints.frame_count_modulo));
      const rem = Number(constraints.frame_count_remainder);
      const k = Math.max(1, Math.round((target - rem) / mod));
      return Math.max(rem + mod, k * mod + rem);
    }
    return target;
  }

  function qualityPresetConfig(value) {
    const presets = {
      preview: { areaScale: 0.75, stepScale: 0.65 },
      balanced: { areaScale: 1.0, stepScale: 1.0 },
      high: { areaScale: 1.2, stepScale: 1.3 },
    };
    return presets[String(value || "balanced")] || presets.balanced;
  }

  function deriveDimensionsFromAspect(sourceWidth, sourceHeight, targetArea, constraints) {
    const aspect = Number(sourceWidth) / Number(sourceHeight || 1);
    const multiple = Math.max(1, Number(constraints.dimension_multiple || 1));
    if (!Number.isFinite(aspect) || aspect <= 0) return null;
    const area = Math.max(256 * 256, Number(targetArea) || 256 * 256);
    const rawWidth = Math.sqrt(area * aspect);
    const rawHeight = rawWidth / aspect;
    return {
      width: snap(rawWidth, multiple),
      height: snap(rawHeight, multiple),
    };
  }

  function updatePresetSummary() {
    if (!activeModel) {
      setText("preset-summary", "Select a model to calculate duration and preset settings.");
      return;
    }
    const seconds = currentTargetSeconds();
    const fps = Math.max(1, Number(byId("gen-fps")?.value || activeModel.defaults?.fps || 1));
    const frames = Math.max(1, Number(byId("gen-frames")?.value || 1));
    const actual = Math.max(0, frames - 1) / fps;
    const width = Number(byId("gen-width")?.value || 0);
    const height = Number(byId("gen-height")?.value || 0);
    const quality = String(byId("preset-quality")?.value || "balanced");
    const label = quality.charAt(0).toUpperCase() + quality.slice(1);
    const dims = width && height ? ` • ${width}×${height}` : "";
    setText(
      "preset-summary",
      `${label} • target ${Number(seconds.toFixed(2))}s → ${frames} frames @ ${fps} fps (${actual.toFixed(2)}s)${dims}`,
    );
  }

  function syncDurationToFrames() {
    if (!activeModel) return;
    const defaults = activeModel.defaults || {};
    const constraints = activeModel.constraints || {};
    const seconds = currentTargetSeconds();
    const fps = Math.max(1, Math.round(Number(byId("gen-fps")?.value || defaults.fps || 16)));
    setInput("gen-fps", fps);
    setInput("gen-frames", frameCountForDuration(seconds, fps, constraints));
    updatePresetSummary();
  }

  function syncFramesToDuration() {
    if (!activeModel) return;
    const fps = Math.max(1, Number(byId("gen-fps")?.value || activeModel.defaults?.fps || 1));
    const frames = Math.max(1, Number(byId("gen-frames")?.value || 1));
    const seconds = Math.max(0.1, (frames - 1) / fps);
    setInput("preset-seconds", Number(seconds.toFixed(2)));
    updatePresetSummary();
  }

  function applyPresetDimensions(preset) {
    if (!activeModel) return;
    const defaults = activeModel.defaults || {};
    const constraints = activeModel.constraints || {};
    const currentWidth = Number(byId("gen-width")?.value || defaults.width || 832);
    const currentHeight = Number(byId("gen-height")?.value || defaults.height || 480);
    const baseArea = Math.max(256 * 256, Number(defaults.width || currentWidth) * Number(defaults.height || currentHeight));
    const targetArea = baseArea * preset.areaScale;

    const applyAspect = (sourceWidth, sourceHeight) => {
      const dims = deriveDimensionsFromAspect(sourceWidth, sourceHeight, targetArea, constraints);
      if (!dims) return;
      setInput("gen-width", dims.width);
      setInput("gen-height", dims.height);
      generateDimsTouched = false;
      updatePresetSummary();
    };

    if (selectedSource?.web_path) {
      const image = new Image();
      image.onload = () => applyAspect(image.naturalWidth, image.naturalHeight);
      image.src = selectedSource.web_path;
      return;
    }
    applyAspect(currentWidth, currentHeight);
  }

  function applyGeneratePreset() {
    if (!activeModel) return;
    const quality = String(byId("preset-quality")?.value || "balanced");
    const preset = qualityPresetConfig(quality);
    const defaults = activeModel.defaults || {};
    const constraints = activeModel.constraints || {};
    const fps = Math.max(1, Math.round(Number(defaults.fps || byId("gen-fps")?.value || 16)));
    setInput("gen-fps", fps);
    setInput("gen-frames", frameCountForDuration(currentTargetSeconds(), fps, constraints));

    if (!stepsLocked(constraints)) {
      const baseSteps = Math.max(1, Number(defaults.num_inference_steps || byId("gen-steps")?.value || 30));
      setInput("gen-steps", Math.max(1, Math.round(baseSteps * preset.stepScale)));
    }
    if (!guidanceLocked(constraints) && defaults.guidance_scale != null) {
      setInput("gen-guidance", defaults.guidance_scale);
    }
    applyPresetDimensions(preset);
    updatePresetSummary();
  }

  function collectGenerationPayload() {
    if (!activeModel) throw new Error("Select a video model first.");
    const caps = activeModel.capabilities || {};
    const constraints = activeModel.constraints || {};
    const defaults = activeModel.defaults || {};
    const prompt = String(byId("prompt")?.value || "").trim();
    if (!prompt) throw new Error("Prompt is required.");
    if (caps.source_image_required && !selectedSource) throw new Error("The selected model requires a source image.");
    if (selectedSource && !caps.image_to_video) throw new Error("The selected model does not support image conditioning.");

    const multiple = Math.max(1, Number(constraints.dimension_multiple || 1));
    const payload = {
      prompt,
      width: snap(byId("gen-width")?.value || defaults.width || 832, multiple),
      height: snap(byId("gen-height")?.value || defaults.height || 480, multiple),
      num_frames: normalizeFrames(byId("gen-frames")?.value || defaults.num_frames || 81, constraints),
      fps: Math.max(1, Math.floor(Number(byId("gen-fps")?.value || defaults.fps || 16))),
    };

    if (selectedSource) {
      payload.image_path = selectedSource.path;
      const stabilityModes = Array.isArray(constraints.i2v_stability_modes)
        ? constraints.i2v_stability_modes.map(String)
        : [];
      const stability = String(byId("i2v-stability")?.value || "").trim();
      if (stability && stabilityModes.includes(stability)) payload.i2v_stability = stability;
    }
    if (caps.negative_prompt) payload.negative_prompt = String(byId("negative-prompt")?.value || "");
    if (!stepsLocked(constraints)) {
      payload.num_inference_steps = Math.max(1, Math.floor(Number(byId("gen-steps")?.value || defaults.num_inference_steps || 30)));
    }
    if (!guidanceLocked(constraints)) {
      payload.guidance_scale = Number(byId("gen-guidance")?.value ?? defaults.guidance_scale ?? 5.0);
    }
    const seedText = String(byId("gen-seed")?.value || "").trim();
    if (seedText !== "") payload.seed = Number(seedText);
    return payload;
  }

  async function loadConfig() {
    const payload = await api.get("/config");
    cachedConfig = payload?.config || {};
    setInput("model-id-or-path", cachedConfig.model_id_or_path || "");
    setInput("models-dir", cachedConfig.models_dir || "");
    setInput("output-dir", cachedConfig.output_dir || "");
    return cachedConfig;
  }

  async function saveConfig({ quiet = false } = {}) {
    const payload = {
      model_id_or_path: String(byId("model-id-or-path")?.value || "").trim(),
      models_dir: String(byId("models-dir")?.value || "").trim(),
      output_dir: String(byId("output-dir")?.value || "").trim(),
    };
    if (!quiet) setBusy("save-config", true, "Saving…");
    try {
      const result = await api.postJson("/config", payload);
      cachedConfig = result?.config || payload;
      if (!quiet) setStatus("ready", "Saved");
      await Promise.all([loadModels(), loadHealth()]);
      return result;
    } finally {
      if (!quiet) setBusy("save-config", false);
    }
  }

  async function loadModels() {
    const payload = await api.get("/models");
    discoveredModels = Array.isArray(payload?.items) ? payload.items : [];
    renderModelSelect();
    return payload;
  }

  async function onModelSelected() {
    const select = byId("discovered-models");
    const index = Number(select?.value);
    activeModel = select?.value !== "" && Number.isInteger(index) && index >= 0 ? discoveredModels[index] || null : null;
    if (activeModel) {
      setInput("model-id-or-path", modelPersistedSource(activeModel));
      applyCapabilityUI({ applyDefaults: true });
      await saveConfig({ quiet: true });
    } else {
      renderModelProfile();
      updateGenerateAvailability();
    }
  }

  function renderReadiness(health) {
    const ready = health?.ready || {};
    const runtime = health?.runtime || {};
    const badges = [
      ["Model selected", !!ready.model_selected],
      ["Model runtime", !!ready.model_runnable],
      ["Host runtime", !!ready.runtime_ready],
      ["Output", !!ready.output_ready],
      ["Ready", !!ready.engine_ready],
    ];
    const badgeWrap = byId("readiness-badges");
    if (badgeWrap) {
      badgeWrap.innerHTML = badges.map(([label, ok]) => `
        <div class="badge ${ok ? "ok" : "bad"}"><strong>${escapeHtml(label)}</strong><span>${ok ? "Ready" : "Not ready"}</span></div>`).join("");
    }

    const summary = byId("readiness-summary");
    if (summary) {
      summary.className = `callout ${ready.engine_ready ? "" : "muted"}`.trim();
      summary.textContent = ready.engine_ready
        ? "Selected model and output path are ready."
        : (health?.reasons || []).join(" • ") || "DuckMotion is not ready yet.";
    }

    const profile = runtime?.profile || {};
    const runtimeWrap = byId("runtime-summary");
    if (runtimeWrap) {
      const entries = [
        ["Device", profile.device || "--"],
        ["Dtype", profile.dtype || "--"],
        ["GPU", profile.cuda_device_name || "--"],
        ["VRAM", profile.total_vram_gb != null ? `${profile.total_vram_gb} GB` : "--"],
      ];
      runtimeWrap.innerHTML = entries.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
    }

    const list = byId("missing-list");
    if (list) {
      const reasons = Array.isArray(health?.reasons) ? health.reasons : [];
      list.innerHTML = reasons.length
        ? reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")
        : "<li>No readiness warnings.</li>";
    }
    setStatus(ready.engine_ready ? "ready" : "warning", ready.engine_ready ? "Ready" : "Needs attention");
  }

  async function loadHealth() {
    const health = await api.get("/health");
    renderReadiness(health);
    if (health?.selected_model) {
      const match = findConfiguredModel(health.selected_model.source || cachedConfig?.model_id_or_path);
      if (match) activeModel = match;
    }
    renderModelProfile();
    applyCapabilityUI();
    return health;
  }

  function mediaCard(item, { recent = false } = {}) {
    const image = item.web_path
      ? `<img class="media-thumb" src="${escapeHtml(item.web_path)}" alt="${escapeHtml(item.name || "image")}" loading="lazy" />`
      : "";
    const action = recent
      ? `<button class="btn btn-secondary btn-sm" data-stage-recent="${escapeHtml(item.path || "")}">Stage</button>`
      : `<button class="btn btn-primary btn-sm" data-select-stage="${escapeHtml(item.path || "")}">Use</button>`;
    const remove = recent ? "" : `<button class="btn btn-secondary btn-sm" data-delete-stage="${escapeHtml(item.name || "")}">Delete</button>`;
    return `<div class="media-entry">${image}<div class="media-entry-copy"><strong>${escapeHtml(item.name || "image")}</strong><span>${escapeHtml(item.run || "")}</span></div><div class="row">${action}${remove}</div></div>`;
  }

  async function loadStaging() {
    const payload = await api.get("/staging");
    const items = Array.isArray(payload?.items) ? payload.items : [];
    const wrap = byId("staging-list");
    if (wrap) wrap.innerHTML = items.length ? items.map((item) => mediaCard(item)).join("") : '<div class="empty-state">No staged images yet.</div>';
    wrap?.querySelectorAll("[data-select-stage]").forEach((button) => {
      button.addEventListener("click", () => {
        selectedSource = items.find((item) => item.path === button.dataset.selectStage) || null;
        renderSelectedSource();
        updatePresetSummary();
        updateGenerateAvailability();
      });
    });
    wrap?.querySelectorAll("[data-delete-stage]").forEach((button) => {
      button.addEventListener("click", async () => {
        await api.del(`/staging/${encodeURIComponent(button.dataset.deleteStage)}`);
        if (selectedSource?.name === button.dataset.deleteStage) clearSelectedSource();
        await loadStaging();
        updateGenerateAvailability();
      });
    });
    return payload;
  }

  async function loadRecent() {
    const payload = await api.get("/webbduck/recent-images?limit=30");
    const items = Array.isArray(payload?.items) ? payload.items : [];
    const wrap = byId("recent-list");
    if (wrap) wrap.innerHTML = items.length ? items.map((item) => mediaCard(item, { recent: true })).join("") : '<div class="empty-state">No recent WebbDuck images found.</div>';
    wrap?.querySelectorAll("[data-stage-recent]").forEach((button) => {
      button.addEventListener("click", async () => {
        const form = new FormData();
        form.append("path", button.dataset.stageRecent || "");
        const result = await api.postForm("/staging/from-webbduck", form);
        selectedSource = result?.item || null;
        renderSelectedSource();
        await loadStaging();
        updatePresetSummary();
        updateGenerateAvailability();
      });
    });
    return payload;
  }

  async function uploadSource(event) {
    event.preventDefault();
    const file = byId("upload-file")?.files?.[0];
    if (!file) return;
    setBusy("upload-btn", true, "Uploading…");
    try {
      const form = new FormData();
      form.append("image", file);
      const result = await api.postForm("/staging/upload", form);
      selectedSource = result?.item || null;
      renderSelectedSource();
      await loadStaging();
      updatePresetSummary();
      updateGenerateAvailability();
    } finally {
      setBusy("upload-btn", false);
    }
  }

  async function submitGeneration() {
    const feedback = byId("generate-feedback");
    try {
      const payload = collectGenerationPayload();
      setBusy("submit-generate", true, "Queueing…");
      const result = await api.postJson("/engine/generate", payload);
      if (feedback) {
        feedback.dataset.jobFeedback = "1";
        feedback.className = "callout";
        feedback.textContent = `Queued ${result?.job?.job_id || "video job"}.`;
      }
      switchView("jobs");
      await loadJobs();
    } catch (error) {
      if (feedback) {
        feedback.dataset.jobFeedback = "1";
        feedback.className = "callout";
        feedback.textContent = error.message || String(error);
      }
      setStatus("error", "Generation error");
    } finally {
      setBusy("submit-generate", false);
      updateGenerateAvailability();
    }
  }

  function jobMarkup(job) {
    const progress = job?.progress || {};
    const status = String(job?.status || "unknown");
    const modelName = job?.model?.name || "Video model";
    const prompt = job?.params?.prompt || "";
    const cancel = ["queued", "running"].includes(status)
      ? `<button class="btn btn-secondary btn-sm" data-cancel-job="${escapeHtml(job.job_id)}">Cancel</button>`
      : "";
    const error = job?.error ? `<div class="callout">${escapeHtml(job.error)}</div>` : "";
    return `<div class="job-card">
      <div class="row job-card-head"><strong>${escapeHtml(modelName)}</strong><span>${escapeHtml(status)}</span></div>
      <div class="hint-inline">${escapeHtml(prompt)}</div>
      <div class="job-progress"><span style="width:${Math.max(0, Math.min(100, Number(progress.percent || 0)))}%"></span></div>
      <div class="row"><span class="hint-inline">${escapeHtml(progress.stage || status)} · ${Number(progress.percent || 0)}%</span>${cancel}</div>
      ${error}
    </div>`;
  }

  async function loadJobs() {
    const payload = await api.get("/engine/jobs?limit=100");
    const jobs = Array.isArray(payload?.jobs) ? payload.jobs : [];
    const wrap = byId("jobs-list");
    if (wrap) wrap.innerHTML = jobs.length ? jobs.map(jobMarkup).join("") : '<div class="empty-state">No DuckMotion jobs yet.</div>';
    wrap?.querySelectorAll("[data-cancel-job]").forEach((button) => {
      button.addEventListener("click", async () => {
        await api.postJson("/engine/cancel", { job_id: button.dataset.cancelJob });
        await loadJobs();
      });
    });
    return payload;
  }

  function galleryMarkup(item) {
    const meta = item?.meta || {};
    const modelName = meta?.model?.name || (typeof meta.model === "string" ? meta.model : "Video model");
    const prompt = meta?.params?.prompt || meta?.prompt || "";
    const videoSrc = item.video ? `${API_BASE}${item.video}` : "";
    const posterAttr = item.poster ? ` poster="${escapeHtml(`${API_BASE}${item.poster}`)}"` : "";
    return `<article class="gallery-card">
      <video controls preload="metadata"${posterAttr} src="${escapeHtml(videoSrc)}"></video>
      <div class="gallery-card-copy"><strong>${escapeHtml(modelName)}</strong><span>${escapeHtml(prompt)}</span></div>
    </article>`;
  }

  async function loadGallery() {
    const payload = await api.get("/gallery?limit=100");
    const items = Array.isArray(payload?.items) ? payload.items : [];
    const wrap = byId("gallery-list");
    if (wrap) wrap.innerHTML = items.length ? items.map(galleryMarkup).join("") : '<div class="empty-state">No generated videos yet.</div>';
    return payload;
  }

  async function unloadModels() {
    setBusy("unload-engine", true, "Unloading…");
    try {
      await api.postJson("/engine/unload", {});
      setStatus("ready", "Models unloaded");
      await loadHealth();
    } finally {
      setBusy("unload-engine", false);
    }
  }

  async function clearJobs() {
    await api.postJson("/jobs/clear", {});
    await loadJobs();
  }

  async function refreshAll() {
    setBusy("refresh-all", true, "Refreshing…");
    try {
      await loadConfig();
      await loadModels();
      await Promise.all([loadHealth(), loadJobs(), loadGallery(), loadStaging(), loadRecent()]);
    } catch (error) {
      setStatus("error", error.message || "Refresh failed");
    } finally {
      setBusy("refresh-all", false);
    }
  }

  function bindEvents() {
    qsa(".tab").forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
    byId("refresh-all")?.addEventListener("click", refreshAll);
    byId("refresh-models")?.addEventListener("click", loadModels);
    byId("discovered-models")?.addEventListener("change", onModelSelected);
    byId("save-config")?.addEventListener("click", () => saveConfig());
    byId("check-health")?.addEventListener("click", loadHealth);
    byId("unload-engine")?.addEventListener("click", unloadModels);
    byId("upload-form")?.addEventListener("submit", uploadSource);
    byId("refresh-staging")?.addEventListener("click", loadStaging);
    byId("refresh-recent")?.addEventListener("click", loadRecent);
    byId("submit-generate")?.addEventListener("click", submitGeneration);
    byId("fill-defaults")?.addEventListener("click", () => applyModelDefaults(true));
    byId("apply-generate-preset")?.addEventListener("click", applyGeneratePreset);
    byId("preset-quality")?.addEventListener("change", applyGeneratePreset);
    byId("preset-seconds")?.addEventListener("input", syncDurationToFrames);
    byId("gen-fps")?.addEventListener("input", syncDurationToFrames);
    byId("gen-frames")?.addEventListener("input", syncFramesToDuration);
    byId("gen-width")?.addEventListener("input", () => { generateDimsTouched = true; updatePresetSummary(); });
    byId("gen-height")?.addEventListener("input", () => { generateDimsTouched = true; updatePresetSummary(); });
    byId("refresh-jobs")?.addEventListener("click", loadJobs);
    byId("clear-jobs")?.addEventListener("click", clearJobs);
    byId("refresh-gallery")?.addEventListener("click", loadGallery);
  }

  async function initialize() {
    bindEvents();
    try {
      await loadConfig();
      await loadModels();
      await Promise.all([loadHealth(), loadStaging(), loadRecent(), loadJobs(), loadGallery()]);
      renderSelectedSource();
      updatePresetSummary();
      updateGenerateAvailability();
    } catch (error) {
      setStatus("error", error.message || "DuckMotion failed to initialize");
    }
    autoTimer = window.setInterval(async () => {
      try {
        await Promise.all([loadHealth(), loadJobs()]);
      } catch (_error) {
        // Keep the UI usable during transient runtime changes.
      }
    }, AUTO_REFRESH_MS);
  }

  window.addEventListener("beforeunload", () => {
    if (autoTimer) window.clearInterval(autoTimer);
  });

  window.addEventListener("message", async (event) => {
    const data = event.data;
    if (!data || data.type !== "webbduck.duckmotion.handoff") return;

    const imagePath = data.image?.src;
    if (!imagePath) return;

    try {
      const form = new FormData();
      form.append("path", imagePath);
      const result = await api.postForm("/staging/from-webbduck", form);
      selectedSource = result?.item || null;
      renderSelectedSource();
      await loadStaging();
      updatePresetSummary();
      updateGenerateAvailability();
    } catch (error) {
      console.error("DuckMotion: failed to stage image from WebbDuck:", error);
    }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();