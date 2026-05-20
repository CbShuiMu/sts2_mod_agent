const API = "";

const fallbackDomains = [
  { id: "cards", label: "Cards", available: true },
  { id: "powers", label: "Powers", available: true },
  { id: "relics", label: "Relics", available: true },
  { id: "potions", label: "Potions", available: true },
  { id: "orbs", label: "Orbs", available: true },
  { id: "enchantments", label: "Enchantments", available: true },
  { id: "afflictions", label: "Afflictions", available: true },
  { id: "rest_site_ui", label: "Rest Site UI", available: true },
  { id: "events", label: "Events", available: true },
];

const state = {
  domains: [],
  lastResult: null,
  requestedDomains: [],
  loading: false,
};

function formatQueryDuration(ms) {
  const value = Number(ms) || 0;
  if (value < 1000) return `${value} ms`;
  if (value < 60000) return `${(value / 1000).toFixed(1)} s`;
  const minutes = Math.floor(value / 60000);
  const seconds = Math.round((value % 60000) / 1000);
  return seconds === 0 ? `${minutes} m` : `${minutes} m ${seconds} s`;
}

const els = {
  form: document.getElementById("queryForm"),
  queryInput: document.getElementById("queryInput"),
  topKInput: document.getElementById("topKInput"),
  contextNInput: document.getElementById("contextNInput"),
  serverStatus: document.getElementById("serverStatus"),
  searchBtn: document.getElementById("searchBtn"),
  libraryGrid: document.getElementById("libraryGrid"),
  selectAllBtn: document.getElementById("selectAllBtn"),
  selectNoneBtn: document.getElementById("selectNoneBtn"),
  summaryStrip: document.getElementById("summaryStrip"),
  summaryText: document.getElementById("summaryText"),
  copyJsonBtn: document.getElementById("copyJsonBtn"),
  results: document.getElementById("results"),
  libraryTemplate: document.getElementById("libraryTemplate"),
  groupTemplate: document.getElementById("groupTemplate"),
  contextTemplate: document.getElementById("contextTemplate"),
};

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

async function boot() {
  bindEvents();
  renderLibraries(fallbackDomains);
  try {
    const health = await request("/api/health");
    const domains = Array.isArray(health.vector_domains) && health.vector_domains.length
      ? health.vector_domains
      : fallbackDomains;
    state.domains = domains;
    els.topKInput.value = health.default_desc_top_k || 4;
    els.contextNInput.value = health.default_context_n || 3;
    const usesMilvusServer = /^https?:\/\//i.test(health.milvus_uri || "");
    els.serverStatus.textContent = health.description_db_exists
      ? "Online"
      : usesMilvusServer
        ? "Milvus collection missing"
        : "Using local Lite DB";
    els.serverStatus.className = `status-pill ${health.description_db_exists ? "online" : "offline"}`;
    renderLibraries(domains);
  } catch (error) {
    state.domains = fallbackDomains;
    els.serverStatus.textContent = `Offline: ${error.message}`;
    els.serverStatus.className = "status-pill offline";
  }
}

function bindEvents() {
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    runQuery();
  });
  els.selectAllBtn.addEventListener("click", () => {
    setLibrarySelection(true);
  });
  els.selectNoneBtn.addEventListener("click", () => {
    setLibrarySelection(false);
  });
  els.copyJsonBtn.addEventListener("click", copyLastJson);
  els.queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      runQuery();
    }
  });
}

function renderLibraries(domains) {
  els.libraryGrid.innerHTML = "";
  for (const domain of domains) {
    const node = els.libraryTemplate.content.firstElementChild.cloneNode(true);
    const checkbox = node.querySelector("input");
    checkbox.value = domain.id;
    checkbox.checked = Boolean(domain.available);
    checkbox.disabled = !domain.available;
    node.classList.toggle("unavailable", !domain.available);
    node.classList.toggle("selected", checkbox.checked);
    checkbox.addEventListener("change", () => {
      node.classList.toggle("selected", checkbox.checked);
    });
    node.querySelector(".library-name").textContent = domain.label || titleCase(domain.id);
    node.querySelector(".library-meta").textContent = domain.available
      ? compactCollection(domain.collection || domain.persist_path || "Ready")
      : "Not built";
    els.libraryGrid.appendChild(node);
  }
}

function setLibrarySelection(selected) {
  els.libraryGrid.querySelectorAll("input[type='checkbox']").forEach((input) => {
    if (!input.disabled) input.checked = selected;
    const tile = input.closest(".library-tile");
    if (tile) tile.classList.toggle("selected", input.checked);
  });
}

function selectedDomains() {
  return Array.from(els.libraryGrid.querySelectorAll("input[type='checkbox']:checked"))
    .map((input) => input.value)
    .filter(Boolean);
}

async function runQuery() {
  const query = els.queryInput.value.trim();
  const domains = selectedDomains();
  if (!query) {
    renderEmpty("Give me a query first.", "Describe the text, class, or behavior you want to retrieve.");
    return;
  }
  if (!domains.length) {
    renderEmpty("Select at least one vector library.", "Use All if you want to search everything currently available.");
    return;
  }

  state.loading = true;
  state.requestedDomains = domains;
  els.searchBtn.disabled = true;
  els.searchBtn.textContent = "Searching";
  renderLoading(query, domains);

  try {
    const result = await request("/api/query", {
      method: "POST",
      body: JSON.stringify({
        query,
        domains,
        desc_top_k: Number(els.topKInput.value || 4),
        context_n: Number(els.contextNInput.value || 3),
      }),
    });
    state.lastResult = result;
    renderResults(result);
  } catch (error) {
    renderEmpty("Query failed.", error.message);
  } finally {
    state.loading = false;
    els.searchBtn.disabled = false;
    els.searchBtn.textContent = "Search";
  }
}

function renderLoading(query, domains) {
  els.summaryStrip.hidden = false;
  els.summaryText.textContent = `Searching ${domains.join(", ")} for "${query}"...`;
  els.results.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.innerHTML = "<div><strong>Searching...</strong><span>Milvus is collecting nearest matches.</span></div>";
  els.results.appendChild(empty);
}

function renderResults(result) {
  const requested = Array.isArray(result.requested_domains) && result.requested_domains.length
    ? result.requested_domains
    : state.requestedDomains;
  const searchPlan = Array.isArray(result.search_plan) && result.search_plan.length
    ? result.search_plan.map((item) => item.vector_db === "descriptions" ? `${item.domain}@descriptions` : item.domain)
    : result.searched_domains || [];
  const groups = filterGroupsByDomains(Array.isArray(result.groups) ? result.groups : [], requested);
  els.summaryStrip.hidden = false;
  els.summaryText.textContent = [
    `${countContexts(groups)} contexts`,
    `${groups.length} query groups`,
    formatQueryDuration(result.duration_ms),
    `requested: ${requested.join(", ") || "none"}`,
    `searched: ${searchPlan.join(", ") || "none"}`,
  ].join(" | ");
  els.results.innerHTML = "";

  const visibleGroups = groups.filter((group) => Array.isArray(group.contexts) && group.contexts.length);
  if (!visibleGroups.length) {
    renderEmpty("No matches found.", "Try a shorter game-text style query or a different vector library.");
    return;
  }

  for (const group of visibleGroups) {
    els.results.appendChild(renderGroup(group));
  }
}

function filterGroupsByDomains(groups, domains) {
  const allowed = new Set((domains || []).map((domain) => String(domain || "").toLowerCase()));
  if (!allowed.size) return groups;
  return groups
    .map((group) => {
      const groupSource = String(group.source_db || "").toLowerCase();
      const contexts = (Array.isArray(group.contexts) ? group.contexts : []).filter((context) => {
        const contextSource = String(context.source_db || "").toLowerCase();
        const contextDomain = String(context.domain || "").toLowerCase();
        return allowed.has(contextSource) || allowed.has(contextDomain) || allowed.has(groupSource);
      });
      return { ...group, contexts };
    })
    .filter((group) => group.contexts.length);
}

function countContexts(groups) {
  return groups.reduce((total, group) => total + (Array.isArray(group.contexts) ? group.contexts.length : 0), 0);
}

function renderGroup(group) {
  const node = els.groupTemplate.content.firstElementChild.cloneNode(true);
  const contexts = Array.isArray(group.contexts) ? group.contexts : [];
  node.querySelector(".group-domain").textContent = group.source_db || "descriptions";
  node.querySelector(".group-query").textContent = group.query || "";
  node.querySelector(".group-count").textContent = `${contexts.length} contexts`;
  const list = node.querySelector(".context-list");
  for (const context of contexts) {
    list.appendChild(renderContext(context));
  }
  return node;
}

function renderContext(context) {
  const node = els.contextTemplate.content.firstElementChild.cloneNode(true);
  const title = context.title || context.entity_name || context.id || "Untitled";
  const filePath = context.mcp_file_path || context.file_path || context.absolute_file_path || "";
  node.querySelector("h3").textContent = title;
  node.querySelector(".score").textContent = context.desc_score ? `Score ${context.desc_score}` : "Score -";
  node.querySelector(".description").textContent = context.description || "No public description metadata.";
  node.querySelector(".domain").textContent = context.source_db || context.domain || "";
  node.querySelector(".role").textContent = context.member_name || context.chunk_type || "";
  node.querySelector(".file").textContent = filePath;
  node.querySelector("code").textContent = context.code || "No code preview returned.";
  return node;
}

function renderEmpty(title, detail) {
  els.summaryStrip.hidden = true;
  els.results.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty-state";
  const content = document.createElement("div");
  const strong = document.createElement("strong");
  const span = document.createElement("span");
  strong.textContent = title;
  span.textContent = detail;
  content.append(strong, span);
  empty.appendChild(content);
  els.results.appendChild(empty);
}

async function copyLastJson() {
  if (!state.lastResult) return;
  const text = JSON.stringify(state.lastResult, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    els.copyJsonBtn.textContent = "Copied";
  } catch {
    els.copyJsonBtn.textContent = "Copy failed";
  }
  setTimeout(() => {
    els.copyJsonBtn.textContent = "Copy JSON";
  }, 900);
}

function compactCollection(value) {
  const text = String(value || "");
  if (text.length <= 32) return text;
  return `${text.slice(0, 14)}...${text.slice(-14)}`;
}

function titleCase(value) {
  return String(value || "")
    .split(/[_-]/)
    .filter(Boolean)
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

boot();
