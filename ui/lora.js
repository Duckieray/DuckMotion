(function () {
  const API_BASE = window.location.pathname.replace(/\/ui\/.*$/, "/api");
  const originalFetch = window.fetch.bind(window);
  let catalog = [];
  const selected = new Map();

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function panel() {
    return document.getElementById("lora-group");
  }

  function ensurePanel() {
    if (panel()) return panel();
    const anchor = document.getElementById("constraint-summary");
    if (!anchor) return null;

    const node = document.createElement("div");
    node.id = "lora-group";
    node.className = "stack hidden";
    node.innerHTML = `
      <div class="row">
        <div class="select-grow">
          <div class="label">LoRAs</div>
          <div class="hint">Shared WebbDuck library · <code>lora/ltx/</code></div>
        </div>
        <button id="refresh-loras" class="btn btn-secondary btn-sm" type="button">Refresh</button>
      </div>
      <div id="lora-list" class="stack"></div>
    `;
    anchor.insertAdjacentElement("afterend", node);
    node.querySelector("#refresh-loras")?.addEventListener("click", () => loadLoras());
    return node;
  }

  function currentSelection() {
    return catalog.flatMap((item, index) => {
      const checkbox = document.querySelector(`[data-lora-enabled="${index}"]`);
      if (!checkbox?.checked) return [];
      const weightInput = document.querySelector(`[data-lora-weight="${index}"]`);
      const weight = Number(weightInput?.value ?? item.weight ?? 1.0);
      return [{ name: String(item.name), weight: Number.isFinite(weight) ? weight : 1.0 }];
    });
  }

  function captureSelection() {
    currentSelection().forEach((item) => selected.set(item.name, item.weight));
  }

  function render(payload) {
    const group = ensurePanel();
    const list = document.getElementById("lora-list");
    if (!group || !list) return;

    captureSelection();
    const supported = !!payload?.supported;
    group.classList.toggle("hidden", !supported);
    if (!supported) {
      list.innerHTML = "";
      catalog = [];
      return;
    }

    catalog = Array.isArray(payload?.items) ? payload.items : [];
    if (!catalog.length) {
      list.innerHTML = '<div class="callout muted">No LTX LoRAs found. Put <code>.safetensors</code> files in WebbDuck\'s <code>lora/ltx/</code> folder and refresh.</div>';
      return;
    }

    list.innerHTML = catalog.map((item, index) => {
      const name = String(item?.name || `LoRA ${index + 1}`);
      const remembered = selected.get(name);
      const defaultWeight = Number.isFinite(Number(remembered))
        ? Number(remembered)
        : (Number.isFinite(Number(item?.weight)) ? Number(item.weight) : 1.0);
      const checked = selected.has(name) ? " checked" : "";
      const trigger = item?.trigger
        ? `<span class="hint-inline">Trigger: <code>${escapeHtml(item.trigger)}</code></span>`
        : "";
      const description = item?.description
        ? `<span class="hint-inline">${escapeHtml(item.description)}</span>`
        : "";
      return `
        <div class="row">
          <input data-lora-enabled="${index}" type="checkbox"${checked} aria-label="Enable ${escapeHtml(name)}" />
          <div class="select-grow stack" style="gap: 0.2rem;">
            <strong>${escapeHtml(name)}</strong>
            ${trigger}${description}
          </div>
          <label class="label" for="lora-weight-${index}">Weight</label>
          <input id="lora-weight-${index}" data-lora-weight="${index}" class="input" style="max-width: 7rem;" type="number" min="-4" max="4" step="0.05" value="${escapeHtml(defaultWeight)}" />
        </div>`;
    }).join("");

    catalog.forEach((item, index) => {
      const checkbox = document.querySelector(`[data-lora-enabled="${index}"]`);
      const weight = document.querySelector(`[data-lora-weight="${index}"]`);
      checkbox?.addEventListener("change", () => {
        const name = String(item.name);
        if (checkbox.checked) {
          const value = Number(weight?.value ?? item.weight ?? 1.0);
          selected.set(name, Number.isFinite(value) ? value : 1.0);
        } else {
          selected.delete(name);
        }
      });
      weight?.addEventListener("input", () => {
        if (!checkbox?.checked) return;
        const value = Number(weight.value);
        if (Number.isFinite(value)) selected.set(String(item.name), value);
      });
    });
  }

  async function loadLoras() {
    try {
      const response = await originalFetch(`${API_BASE}/loras`, { method: "GET" });
      if (!response.ok) throw new Error(`LoRA catalog request failed (${response.status})`);
      render(await response.json());
    } catch (error) {
      const group = ensurePanel();
      const list = document.getElementById("lora-list");
      if (group) group.classList.remove("hidden");
      if (list) {
        list.innerHTML = `<div class="callout muted">Could not load LoRAs: ${escapeHtml(error?.message || error)}</div>`;
      }
    }
  }

  window.fetch = async function (input, init = {}) {
    const url = typeof input === "string" ? input : String(input?.url || "");
    const method = String(init?.method || (typeof input !== "string" ? input?.method : "GET") || "GET").toUpperCase();
    let nextInit = init;

    if (method === "POST" && url.endsWith("/engine/generate") && typeof init?.body === "string") {
      try {
        const payload = JSON.parse(init.body);
        payload.loras = currentSelection();
        nextInit = { ...init, body: JSON.stringify(payload) };
      } catch (_error) {
        // Leave non-JSON requests untouched; the backend will validate them.
      }
    }

    const response = await originalFetch(input, nextInit);
    if (response.ok && method === "POST" && url.endsWith("/config")) {
      setTimeout(() => loadLoras(), 0);
    }
    return response;
  };

  function initialize() {
    ensurePanel();
    loadLoras();
    document.getElementById("discovered-models")?.addEventListener("change", () => {
      setTimeout(() => loadLoras(), 250);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
