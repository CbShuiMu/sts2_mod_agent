const API = "";
const STORAGE_KEY = "sts2-rag-agent-state-v3";
const i18n = window.STS2_I18N;
const ATTACHMENT_EXTENSIONS = new Set(["json", "txt"]);
const ATTACHMENT_MIME_TYPES = new Set(["application/json", "text/plain"]);

const state = {
  config: null,
  conversations: [],
  activeId: "",
  sending: false,
  sendingId: "",
  abortController: null,
  lang: i18n.defaultLang,
};

const els = {
  serverState: document.getElementById("serverState"),
  newChatBtn: document.getElementById("newChatBtn"),
  conversationList: document.getElementById("conversationList"),
  conversationTitle: document.getElementById("conversationTitle"),
  conversationId: document.getElementById("conversationId"),
  chatMeta: document.getElementById("chatMeta"),
  providerSelect: document.getElementById("providerSelect"),
  modelInput: document.getElementById("modelInput"),
  saveDefaultBtn: document.getElementById("saveDefaultBtn"),
  ragToggle: document.getElementById("ragToggle"),
  agentToggle: document.getElementById("agentToggle"),
  customDomainToggle: document.getElementById("customDomainToggle"),
  customDomainDialog: document.getElementById("customDomainDialog"),
  customDomainList: document.getElementById("customDomainList"),
  customDomainConfirm: document.getElementById("customDomainConfirm"),
  contextN: document.getElementById("contextN"),
  descTopK: document.getElementById("descTopK"),
  fileChips: document.getElementById("fileChips"),
  messages: document.getElementById("messages"),
  composerFiles: document.getElementById("composerFiles"),
  messageInput: document.getElementById("messageInput"),
  sendBtn: document.getElementById("sendBtn"),
  composer: document.querySelector(".composer"),
  chatPanel: document.querySelector(".chat-panel"),
  settingsBtn: document.getElementById("settingsBtn"),
  settingsDialog: document.getElementById("settingsDialog"),
  providerSettings: document.getElementById("providerSettings"),
  fileMcpBtn: document.getElementById("fileMcpBtn"),
  fileDialog: document.getElementById("fileDialog"),
  filePathInput: document.getElementById("filePathInput"),
  listFilesBtn: document.getElementById("listFilesBtn"),
  fileBrowser: document.getElementById("fileBrowser"),
  template: document.getElementById("messageTemplate"),
};

function t(key, params = {}) {
  return i18n.t(state.lang, key, params);
}

function uid() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatDuration(ms) {
  const value = Number(ms) || 0;
  if (value < 1000) return `${value}ms`;
  if (value < 60000) return `${(value / 1000).toFixed(1)}s`;
  const minutes = Math.floor(value / 60000);
  const seconds = Math.round((value % 60000) / 1000);
  return seconds === 0 ? `${minutes}m` : `${minutes}m ${seconds}s`;
}

function newConversation() {
  return {
    id: uid(),
    title: t("newTitle"),
    autoTitle: true,
    messages: [],
    selectedFiles: [],
    draftAttachments: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    providerId: state.config?.default_provider_id || "deepseek",
    model: "",
    memorySummary: "",
  };
}

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    state.conversations = Array.isArray(saved.conversations) ? saved.conversations : [];
    state.activeId = saved.activeId || "";
    state.lang = saved.lang === "en" ? "en" : "zh";
  } catch {
    state.conversations = [];
    state.activeId = "";
    state.lang = i18n.defaultLang;
  }

  if (!state.conversations.length) {
    const conversation = newConversation();
    state.conversations.push(conversation);
    state.activeId = conversation.id;
  }
  if (!state.activeId || !state.conversations.some((item) => item.id === state.activeId)) {
    state.activeId = state.conversations[0].id;
  }
  for (const conversation of state.conversations) {
    conversation.selectedFiles ||= [];
    conversation.draftAttachments ||= [];
    conversation.messages ||= [];
    conversation.memorySummary ||= "";
    conversation.autoTitle ||= conversation.title === "New chat" || conversation.title === "\u65b0\u7684\u4f1a\u8bdd";
    for (const message of conversation.messages) {
      message.id ||= uid();
      if (message.pending && (message.content === "思考中..." || message.content === "Thinking...")) {
        message.content = "";
      }
    }
  }
}

function saveState() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      conversations: state.conversations,
      activeId: state.activeId,
      lang: state.lang,
    }),
  );
}

function activeConversation() {
  return state.conversations.find((item) => item.id === state.activeId);
}

function createConversation() {
  const conversation = newConversation();
  state.conversations.unshift(conversation);
  state.activeId = conversation.id;
  saveState();
  render();
}

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
  loadState();
  bindEvents();
  render();
  try {
    const [health, config] = await Promise.all([request("/api/health"), request("/api/config")]);
    state.config = config;
    els.serverState.dataset.statusKey = health.description_db_exists ? "online" : "noVectorDb";
    els.contextN.value = health.default_context_n || 3;
    els.descTopK.value = health.default_desc_top_k || 4;
    for (const conversation of state.conversations) {
      conversation.providerId ||= config.default_provider_id;
    }
    render();
  } catch (error) {
    els.serverState.dataset.statusKey = "offline";
    appendLocalError(t("backendOffline", { message: error.message }));
  }
}

function bindEvents() {
  els.newChatBtn.addEventListener("click", createConversation);
  els.settingsBtn.addEventListener("click", () => {
    renderSettings();
    els.settingsDialog.showModal();
  });
  els.fileMcpBtn.addEventListener("click", () => {
    els.fileDialog.showModal();
    listFiles(els.filePathInput.value || ".");
  });
  els.listFilesBtn.addEventListener("click", () => listFiles(els.filePathInput.value || "."));
  els.sendBtn.addEventListener("click", handleSendButtonClick);
  els.saveDefaultBtn.addEventListener("click", saveDefaultProvider);
  els.providerSelect.addEventListener("change", () => {
    const conversation = activeConversation();
    if (!conversation) return;
    conversation.providerId = els.providerSelect.value;
    const provider = state.config?.providers?.[conversation.providerId];
    conversation.model = provider?.model || "";
    els.modelInput.value = conversation.model;
    saveState();
    renderMeta();
  });
  els.modelInput.addEventListener("change", () => {
    const conversation = activeConversation();
    if (!conversation) return;
    conversation.model = els.modelInput.value.trim();
    saveState();
    renderMeta();
  });
  els.conversationTitle.addEventListener("change", () => {
    const conversation = activeConversation();
    if (!conversation) return;
    conversation.title = els.conversationTitle.value.trim() || t("unnamedTitle");
    conversation.autoTitle = false;
    conversation.updatedAt = new Date().toISOString();
    saveState();
    renderConversationList();
  });
  els.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  els.messageInput.addEventListener("input", () => {
    resizeComposer();
    updateSendButton();
  });
  setupComposerDropZone();
}

function setupComposerDropZone() {
  const zones = [els.chatPanel, els.composer, els.messageInput].filter(Boolean);
  if (!zones.length) return;

  const isFileDrag = (event) =>
    Array.from(event.dataTransfer?.types || []).includes("Files");

  let dragDepth = 0;
  const setDragging = (on) => {
    if (els.chatPanel) els.chatPanel.classList.toggle("chat-panel-dragover", on);
    if (els.composer) els.composer.classList.toggle("composer-dragover", on);
  };

  window.addEventListener("dragenter", (event) => {
    if (!isFileDrag(event)) return;
    dragDepth += 1;
    setDragging(true);
  });
  window.addEventListener("dragleave", (event) => {
    if (!isFileDrag(event)) return;
    if (!event.relatedTarget) {
      dragDepth = 0;
      setDragging(false);
      return;
    }
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) setDragging(false);
  });
  window.addEventListener("drop", () => {
    dragDepth = 0;
    setDragging(false);
  });
  window.addEventListener("dragend", () => {
    dragDepth = 0;
    setDragging(false);
  });

  for (const zone of zones) {
    zone.addEventListener("dragover", (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      event.stopPropagation();
      event.dataTransfer.dropEffect = "copy";
    });
    zone.addEventListener("drop", async (event) => {
      if (!isFileDrag(event)) return;
      event.preventDefault();
      event.stopPropagation();
      dragDepth = 0;
      setDragging(false);
      const files = Array.from(event.dataTransfer.files || []);
      if (files.length) await ingestDroppedFiles(files);
    });
  }

  // Also block default browser navigation when dropped outside the chat panel.
  window.addEventListener("dragover", (event) => {
    if (isFileDrag(event)) event.preventDefault();
  });
  window.addEventListener("drop", (event) => {
    if (isFileDrag(event) && !els.chatPanel?.contains(event.target)) {
      event.preventDefault();
    }
  });
}

async function ingestDroppedFiles(files) {
  const conversation = activeConversation();
  if (!conversation) return;
  conversation.draftAttachments ||= [];
  const rejectedFiles = [];
  let added = 0;
  for (const file of files) {
    const meta = fileMeta(file);
    if (!meta.supported) {
      rejectedFiles.push(meta.name);
      continue;
    }
    try {
      const text = await file.text();
      const key = `${meta.name}:${file.size}:${file.lastModified}`;
      if (conversation.draftAttachments.some((item) => item.key === key)) continue;
      conversation.draftAttachments.push({
        id: uid(),
        key,
        name: meta.name,
        extension: meta.extension,
        type: file.type || `text/${meta.extension}`,
        size: file.size,
        lastModified: file.lastModified || 0,
        content: text,
      });
      added += 1;
    } catch (error) {
      rejectedFiles.push(`${meta.name} (${error.message})`);
    }
  }
  if (added) {
    saveState();
    renderComposerAttachments();
    updateSendButton();
    els.messageInput.focus();
  }
  if (rejectedFiles.length) {
    alert(t("dropUnsupported", { files: rejectedFiles.join(", ") }));
  }
  return;

  const accepted = [];
  const rejected = [];
  for (const file of files) {
    const name = file.name || "";
    const lower = name.toLowerCase();
    const isJson = lower.endsWith(".json") || file.type === "application/json";
    const isTxt = lower.endsWith(".txt") || file.type === "text/plain";
    if (!isJson && !isTxt) {
      rejected.push(name || "unknown");
      continue;
    }
    try {
      const text = await file.text();
      const lang = isJson ? "json" : "";
      accepted.push(`【${name}】\n\`\`\`${lang}\n${text.replace(/```/g, "`​``")}\n\`\`\``);
    } catch (error) {
      rejected.push(`${name} (${error.message})`);
    }
  }

  if (accepted.length) {
    const input = els.messageInput;
    const block = accepted.join("\n\n");
    const before = input.value;
    const sep = before && !before.endsWith("\n") ? "\n\n" : (before ? "\n" : "");
    const start = input.selectionStart ?? before.length;
    const end = input.selectionEnd ?? before.length;
    if (typeof start === "number" && start !== before.length) {
      input.value = before.slice(0, start) + block + "\n" + before.slice(end);
      const cursor = start + block.length + 1;
      input.setSelectionRange(cursor, cursor);
    } else {
      input.value = before + sep + block + "\n";
      input.setSelectionRange(input.value.length, input.value.length);
    }
    input.focus();
    resizeComposer();
  }

  if (rejected.length) {
    alert(t("dropUnsupported", { files: rejected.join(", ") }));
  }
}

function fileMeta(file) {
  const name = file.name || "unknown";
  const extension = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
  return {
    name,
    extension,
    supported: ATTACHMENT_EXTENSIONS.has(extension) || ATTACHMENT_MIME_TYPES.has(file.type),
  };
}

function handleSendButtonClick() {
  const conversation = activeConversation();
  if (state.sending && conversation && state.sendingId === conversation.id) {
    abortStream();
    return;
  }
  if (state.sending) return;
  sendMessage();
}

function abortStream() {
  const conversation = state.conversations.find((item) => item.id === state.sendingId) || activeConversation();
  const pending = conversation?.messages?.findLast?.((item) => item.pending);
  if (pending) {
    pending.pending = false;
    pending.content ||= t("aborted");
    pending.events ||= [];
    pending.events.push({ type: "aborted", message: t("aborted") });
    conversation.updatedAt = new Date().toISOString();
    flushSaveState();
    render();
  }
  state.sending = false;
  state.sendingId = "";
  if (state.abortController) state.abortController.abort();
}

function toggleLanguage() {
  state.lang = state.lang === "zh" ? "en" : "zh";
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  saveState();
  render();
  if (els.settingsDialog.open) renderSettings();
  if (els.fileDialog.open) listFiles(els.filePathInput.value || ".");
}

function applyTranslations() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder));
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
    node.setAttribute("aria-label", t(node.dataset.i18nAria));
  });
  document.querySelectorAll("[data-i18n-title]").forEach((node) => {
    node.setAttribute("title", t(node.dataset.i18nTitle));
  });
  const statusKey = els.serverState.dataset.statusKey;
  if (statusKey) {
    els.serverState.textContent = t(statusKey);
  }
}

function render() {
  applyTranslations();
  renderProviderSelect();
  renderConversationList();
  renderActiveConversation();
  renderFileChips();
  renderComposerAttachments();
  updateSendButton();
}

function updateSendButton() {
  const conversation = activeConversation();
  const isSendingHere = state.sending && conversation && state.sendingId === conversation.id;
  const hasDraft = Boolean(els.messageInput.value.trim() || conversation?.draftAttachments?.length);
  els.sendBtn.disabled = (state.sending && !isSendingHere) || (!state.sending && !hasDraft);
  els.sendBtn.textContent = isSendingHere ? t("stop") : t("send");
}

function renderProviderSelect() {
  if (!state.config) return;
  const conversation = activeConversation();
  const current = conversation?.providerId || state.config.default_provider_id;
  els.providerSelect.innerHTML = "";
  for (const provider of state.config.catalog) {
    const option = document.createElement("option");
    const saved = state.config.providers?.[provider.id];
    option.value = provider.id;
    option.textContent = `${provider.name}${saved?.has_key ? "" : t("unconfigured")}`;
    els.providerSelect.appendChild(option);
  }
  els.providerSelect.value = current;
  const provider = state.config.providers?.[current];
  if (conversation && !conversation.model) {
    conversation.model = provider?.model || "";
  }
  els.modelInput.value = conversation?.model || provider?.model || "";
}

function renderConversationList() {
  els.conversationList.innerHTML = "";
  for (const conversation of state.conversations) {
    const row = document.createElement("div");
    row.className = `conversation-item${conversation.id === state.activeId ? " active" : ""}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-main";
    button.innerHTML = `
      <span class="conversation-title"></span>
      <span class="conversation-time">${formatTime(conversation.updatedAt)}</span>
    `;
    button.querySelector(".conversation-title").textContent = conversation.autoTitle ? t("newTitle") : conversation.title;
    button.addEventListener("click", () => {
      state.activeId = conversation.id;
      saveState();
      render();
    });

    const deleteButton = document.createElement("button");
    deleteButton.className = "conversation-delete";
    deleteButton.type = "button";
    deleteButton.textContent = "x";
    deleteButton.title = t("deleteConversation");
    deleteButton.addEventListener("click", () => deleteConversation(conversation.id));

    row.append(button, deleteButton);
    els.conversationList.appendChild(row);
  }
}

function deleteConversation(id) {
  if (state.conversations.length === 1) {
    state.conversations[0] = newConversation();
    state.activeId = state.conversations[0].id;
  } else {
    state.conversations = state.conversations.filter((item) => item.id !== id);
    if (state.activeId === id) {
      state.activeId = state.conversations[0].id;
    }
  }
  saveState();
  render();
}

function renderActiveConversation() {
  const conversation = activeConversation();
  if (!conversation) return;
  const stickToBottom = isNearMessageBottom();
  els.conversationTitle.value = conversation.autoTitle ? t("newTitle") : conversation.title;
  els.conversationId.textContent = `conversation_id: ${conversation.id}`;
  els.conversationId.title = `data/logs/conversation_${conversation.id}.jsonl`;
  renderMeta();
  els.messages.innerHTML = "";
  if (!conversation.messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = "<div><strong></strong><span></span></div>";
    empty.querySelector("strong").textContent = t("emptyTitle");
    empty.querySelector("span").textContent = t("emptySubtitle");
    els.messages.appendChild(empty);
    return;
  }
  for (const message of conversation.messages) {
    appendMessageNode(message);
  }
  if (stickToBottom) scrollMessages();
}

function renderMeta() {
  const conversation = activeConversation();
  if (!conversation) return;
  const provider = state.config?.providers?.[conversation.providerId];
  const keyState = provider?.has_key ? t("keyReady") : t("noKey");
  els.chatMeta.textContent = t("chatMeta", { count: conversation.messages.length, keyState });
}

function renderFileChips() {
  const conversation = activeConversation();
  const selectedFiles = conversation?.selectedFiles || [];
  els.fileChips.innerHTML = "";
  for (const path of selectedFiles) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = "<span></span><button type=\"button\">x</button>";
    chip.querySelector("span").textContent = path;
    chip.querySelector("button").title = t("remove");
    chip.querySelector("button").addEventListener("click", () => {
      conversation.selectedFiles = conversation.selectedFiles.filter((item) => item !== path);
      saveState();
      renderFileChips();
    });
    els.fileChips.appendChild(chip);
  }
}

function renderComposerAttachments() {
  if (!els.composerFiles) return;
  const conversation = activeConversation();
  const attachments = conversation?.draftAttachments || [];
  els.composerFiles.innerHTML = "";
  els.composerFiles.hidden = !attachments.length;
  for (const attachment of attachments) {
    els.composerFiles.appendChild(renderAttachmentChip(attachment, {
      removable: true,
      onRemove: () => {
        conversation.draftAttachments = conversation.draftAttachments.filter((item) => item.id !== attachment.id);
        saveState();
        renderComposerAttachments();
        updateSendButton();
      },
    }));
  }
}

function renderMessageAttachments(message, bubble) {
  const attachments = Array.isArray(message.attachments) ? message.attachments : [];
  if (!attachments.length) return;
  const wrap = document.createElement("div");
  wrap.className = "message-attachments";
  for (const attachment of attachments) {
    wrap.appendChild(renderAttachmentChip(attachment));
  }
  bubble.appendChild(wrap);
}

function renderAttachmentChip(attachment, options = {}) {
  const chip = document.createElement("div");
  chip.className = "attachment-chip";
  chip.innerHTML = `
    <span class="attachment-icon">FILE</span>
    <span class="attachment-text">
      <span class="attachment-name"></span>
      <span class="attachment-meta"></span>
    </span>
  `;
  chip.querySelector(".attachment-name").textContent = attachment.name || "unknown";
  chip.querySelector(".attachment-meta").textContent = `${(attachment.extension || "txt").toUpperCase()} | ${formatBytes(attachment.size)}`;
  if (options.removable) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "attachment-remove";
    button.textContent = "x";
    button.title = t("remove");
    button.addEventListener("click", options.onRemove);
    chip.appendChild(button);
  }
  return chip;
}

function formatBytes(size) {
  const value = Number(size) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(value < 10 * 1024 ? 1 : 0)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function appendMessageNode(message) {
  const node = els.template.content.firstElementChild.cloneNode(true);
  node.classList.add(message.role);
  if (message.pending) node.classList.add("pending");
  if (message.id) node.dataset.messageId = message.id;
  node.querySelector(".avatar").textContent = message.role === "user" ? t("youAvatar") : message.role === "error" ? "!" : "AI";
  const bubble = node.querySelector(".bubble");
  bubble.textContent = "";
  if (message.reasoningContent) {
    const reasoning = document.createElement("details");
    reasoning.className = "reasoning-panel";
    reasoning.open = Boolean(message.pending);
    const summary = document.createElement("summary");
    summary.textContent = t("reasoningTitle");
    const body = document.createElement("div");
    body.className = "reasoning-body";
    body.textContent = message.reasoningContent;
    reasoning.append(summary, body);
    bubble.appendChild(reasoning);
  }
  const liveStatus = renderLiveAgentStatus(message);
  if (liveStatus) bubble.appendChild(liveStatus);
  renderMessageAttachments(message, bubble);
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = message.displayContent ?? message.content ?? "";
  bubble.appendChild(content);
  const completionNode = renderCompletionBadge(message);
  if (completionNode) bubble.appendChild(completionNode);
  if (message.meta) {
    const meta = document.createElement("div");
    meta.className = "meta-row";
    meta.textContent = message.meta;
    bubble.appendChild(meta);
  }
  renderRagTrace(message, bubble);
  els.messages.appendChild(node);
}

// Visible completion flag rendered on the assistant bubble after the agent
// finishes. Status is derived from the backend `done` event; it tells the user
// whether the task wrapped up cleanly, hit the step budget, or finished with
// zero file writes (which is suspicious for a code-generation request).
// Live progress chip shown on the assistant bubble while the agent is still
// running. Reading from `message.events` we can already tell how many writes
// happened and which step the agent is on, so the user sees concrete progress
// instead of staring at a spinner.
function renderLiveAgentStatus(message) {
  if (!message.pending) return null;
  if (message.role !== "assistant") return null;
  const events = Array.isArray(message.events) ? message.events : [];
  if (!events.length) return null;
  const stats = summarizeRagEvents(events);
  let stepInfo = null;
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev && ev.type === "generation_start" && ev.step && ev.max_steps) {
      stepInfo = { step: ev.step, max: ev.max_steps };
      break;
    }
  }
  const reads = stats.local_file_read + stats.local_file_read_many;
  const writes = stats.local_file_write;
  const replaces = stats.local_file_replace;
  // Skip the chip if the agent literally hasn't done anything visible yet.
  if (stats.tool_total === 0 && !stepInfo) return null;

  const node = document.createElement("div");
  node.className = "live-status";
  const dot = document.createElement("span");
  dot.className = "live-status-dot";
  node.appendChild(dot);
  const parts = [];
  if (stepInfo && stepInfo.step !== "wrap_up") {
    parts.push(t("liveStep", { step: stepInfo.step, max: stepInfo.max }));
  }
  if (writes) parts.push(t("liveWrites", { count: writes }));
  if (replaces) parts.push(t("liveReplaces", { count: replaces }));
  if (reads) parts.push(t("liveReads", { count: reads }));
  if (stats.rag_query) parts.push(t("liveRag", { count: stats.rag_query }));
  const text = document.createElement("span");
  text.textContent = parts.length ? parts.join(" · ") : t("liveWorking");
  node.appendChild(text);
  return node;
}

function renderCompletionBadge(message) {
  const completion = message.completion;
  if (!completion || message.pending) return null;
  if (message.role !== "assistant") return null;

  const status = completion.status || "ok";
  const writes = completion.writes || 0;
  const files = Array.isArray(completion.files) ? completion.files : [];
  const useAgent = !!completion.use_agent;

  // Decide the UI variant. Agent + 0 writes is downgraded to a warning so the
  // user notices something went wrong even when the backend says ok.
  let variant = "ok";
  let icon = "✅";
  let labelKey = "completionOk";
  if (status === "error") {
    variant = "error";
    icon = "❌";
    labelKey = "completionError";
  } else if (status === "aborted") {
    variant = "warn";
    icon = "⏹";
    labelKey = "completionAborted";
  } else if (status === "budget_exhausted") {
    variant = "warn";
    icon = "⚠️";
    labelKey = "completionBudget";
  } else if (useAgent && writes === 0) {
    // Treat "agent ran but wrote nothing" as info — could be intentional
    // (Q&A-only request) but it's worth surfacing. Color is neutral, not red.
    variant = "info";
    icon = "ℹ️";
    labelKey = "completionNoWrites";
  } else if (useAgent && writes > 0) {
    icon = "✅";
    labelKey = "completionOkWithWrites";
  }

  const node = document.createElement("div");
  node.className = `completion-badge completion-${variant}`;
  const head = document.createElement("div");
  head.className = "completion-head";
  const iconNode = document.createElement("span");
  iconNode.className = "completion-icon";
  iconNode.textContent = icon;
  const labelNode = document.createElement("span");
  labelNode.className = "completion-label";
  labelNode.textContent = t(labelKey, { count: writes });
  head.append(iconNode, labelNode);
  if (completion.duration_ms) {
    const time = document.createElement("span");
    time.className = "completion-time";
    time.textContent = formatDuration(completion.duration_ms);
    head.appendChild(time);
  }
  node.appendChild(head);

  if (files.length) {
    const list = document.createElement("ul");
    list.className = "completion-files";
    for (const file of files) {
      const li = document.createElement("li");
      li.textContent = compactPath(String(file || ""), 64);
      li.title = String(file || "");
      list.appendChild(li);
    }
    node.appendChild(list);
  }
  return node;
}

// Tool emoji icon for compact visual recognition.
const TOOL_ICONS = {
  rag_query: "🔎",
  local_file_read: "📖",
  local_file_read_many: "📚",
  local_file_search: "🔍",
  local_file_list: "📁",
  local_file_write: "✏️",
  local_file_replace: "♻️",
  local_file_copy_tree: "📦",
  local_file_create_dir: "🗂️",
  agent_tool_decision: "💭",
};

const EVENT_ICONS = {
  start: "▶",
  memory_loaded: "🧠",
  retrieval_start: "🔎",
  retrieval_done: "🔎",
  local_files_done: "📁",
  prompt_ready: "📝",
  agent_start: "🤖",
  generation_start: "💬",
  summary_start: "🧾",
  memory_updated: "🧾",
  warning: "⚠",
  done: "✅",
  aborted: "⏹",
  error: "❌",
};

// Middle-ellipsis path so users can still see the basename of long paths.
function compactPath(input, max = 56) {
  const text = String(input || "");
  if (text.length <= max) return text;
  const head = Math.max(8, Math.floor(max / 2) - 2);
  const tail = Math.max(8, max - head - 1);
  return `${text.slice(0, head)}…${text.slice(-tail)}`;
}

// Stable key used to coalesce repeated traces (same tool + same target args).
function traceCoalesceKey(event) {
  const tool = event.tool || "";
  const args = event.arguments || {};
  const result = event.result || {};
  const path = result.path || result.target_path || args.path || args.target_path || args.source_path || "";
  if (path) return `${tool}::${path}`;
  if (tool === "local_file_search") return `${tool}::${args.query || ""}@${args.path || ""}`;
  if (tool === "local_file_read_many") return `${tool}::${(args.paths || []).join("|")}`;
  if (tool === "rag_query") return `${tool}::${args.query || result.query || ""}`;
  return `${tool}::${JSON.stringify(args)}`;
}

// Aggregate per-tool counters and timing for the headline summary chips.
function summarizeRagEvents(events) {
  const stats = {
    retrieval: 0,
    rag_query: 0,
    local_file_read: 0,
    local_file_read_many: 0,
    local_file_search: 0,
    local_file_list: 0,
    local_file_write: 0,
    local_file_replace: 0,
    local_file_copy_tree: 0,
    local_file_create_dir: 0,
    cache_hits: 0,
    duration_ms: 0,
    tool_total: 0,
  };
  for (const event of events) {
    if (!event) continue;
    if (event.type === "retrieval_done") stats.retrieval += 1;
    if (event.type === "done") stats.duration_ms = event.duration_ms || stats.duration_ms;
    if (event.type !== "agent_trace") continue;
    const tool = event.tool || "";
    if (Object.prototype.hasOwnProperty.call(stats, tool)) stats[tool] += 1;
    stats.tool_total += 1;
    const result = event.result || {};
    if (tool === "local_file_read" && result.cache_hit) stats.cache_hits += 1;
    if (tool === "local_file_read_many" && Array.isArray(result.items)) {
      stats.cache_hits += result.items.filter((it) => it && it.cache_hit).length;
    }
  }
  return stats;
}

function buildRagSummaryStrip(stats) {
  const chips = [];
  const push = (key, params) => {
    chips.push(t(key, params));
  };
  if (stats.retrieval) push("ragSummaryRetrieval", { count: stats.retrieval });
  if (stats.rag_query) push("ragSummaryRag", { count: stats.rag_query });
  const reads = stats.local_file_read + stats.local_file_read_many;
  if (reads) push("ragSummaryReads", { count: reads });
  if (stats.local_file_search) push("ragSummarySearches", { count: stats.local_file_search });
  if (stats.local_file_write) push("ragSummaryWrites", { count: stats.local_file_write });
  if (stats.local_file_replace) push("ragSummaryReplaces", { count: stats.local_file_replace });
  if (stats.cache_hits) push("ragSummaryCache", { count: stats.cache_hits });
  if (stats.duration_ms) push("ragSummaryDuration", { duration: formatDuration(stats.duration_ms) });
  return chips;
}

// Group events into phases so the user can collapse/expand each block.
function partitionRagEvents(events) {
  const retrieval = [];
  const tools = [];
  const generation = [];
  const RETRIEVAL_TYPES = new Set([
    "memory_loaded",
    "retrieval_start",
    "retrieval_done",
    "local_files_done",
    "prompt_ready",
  ]);
  const GENERATION_TYPES = new Set([
    "generation_start",
    "summary_start",
    "memory_updated",
    "done",
    "aborted",
    "error",
    "warning",
  ]);

  // Coalesce consecutive duplicate agent_traces with the same key.
  let lastToolKey = null;
  let lastToolRow = null;
  for (const event of events) {
    if (!event) continue;
    if (RETRIEVAL_TYPES.has(event.type) || event.type === "start") {
      retrieval.push(event);
      continue;
    }
    if (GENERATION_TYPES.has(event.type)) {
      generation.push(event);
      continue;
    }
    if (event.type === "agent_start") {
      tools.push(event);
      continue;
    }
    if (event.type === "agent_trace") {
      const key = traceCoalesceKey(event);
      if (key === lastToolKey && lastToolRow) {
        lastToolRow.repeats = (lastToolRow.repeats || 1) + 1;
        lastToolRow.event = event; // keep the latest result snapshot
        continue;
      }
      const row = { event, repeats: 1 };
      tools.push(row);
      lastToolKey = key;
      lastToolRow = row;
      continue;
    }
    // Fallback: bucket unknown events into the generation phase.
    generation.push(event);
  }
  return { retrieval, tools, generation };
}

function renderRagTrace(message, bubble) {
  const events = Array.isArray(message.events) ? message.events : [];
  if (!events.length && !message.memorySummary) return;

  const stats = summarizeRagEvents(events);
  const partition = partitionRagEvents(events);

  const panel = document.createElement("details");
  panel.className = "rag-panel";
  // Auto-expand while pending or when interesting things happened.
  if (message.pending || partition.tools.length || stats.retrieval) {
    panel.open = true;
  }

  const summary = document.createElement("summary");
  summary.className = "rag-summary";
  const titleNode = document.createElement("span");
  titleNode.className = "rag-summary-title";
  titleNode.textContent = t("ragTraceTitle");
  summary.appendChild(titleNode);

  const chips = buildRagSummaryStrip(stats);
  if (chips.length) {
    const chipStrip = document.createElement("span");
    chipStrip.className = "rag-summary-chips";
    for (const chip of chips) {
      const node = document.createElement("span");
      node.className = "rag-chip";
      node.textContent = chip;
      chipStrip.appendChild(node);
    }
    summary.appendChild(chipStrip);
  }
  panel.appendChild(summary);

  const renderPhase = (titleKey, items, { defaultOpen = false } = {}) => {
    if (!items.length) return;
    const phase = document.createElement("details");
    phase.className = "rag-phase";
    if (defaultOpen) phase.open = true;
    const phaseSummary = document.createElement("summary");
    phaseSummary.textContent = `${t(titleKey)} · ${items.length}`;
    phase.appendChild(phaseSummary);
    const timeline = document.createElement("div");
    timeline.className = "rag-timeline";
    for (const item of items) {
      const node = item && item.event
        ? renderToolRow(item.event, item.repeats || 1)
        : renderTraceEvent(item);
      if (node) timeline.appendChild(node);
    }
    phase.appendChild(timeline);
    panel.appendChild(phase);
  };

  renderPhase("ragPhaseRetrieval", partition.retrieval, { defaultOpen: false });
  renderPhase("ragPhaseTools", partition.tools, { defaultOpen: true });
  renderPhase("ragPhaseGeneration", partition.generation, { defaultOpen: false });

  if (message.memorySummary) {
    const memory = document.createElement("div");
    memory.className = "rag-memory";
    memory.textContent = `${t("memorySummary")}: ${message.memorySummary}`;
    panel.appendChild(memory);
  }

  bubble.appendChild(panel);
}

function renderToolRow(event, repeats) {
  const row = renderTraceEvent(event);
  if (repeats && repeats > 1) {
    const badge = document.createElement("span");
    badge.className = "rag-badge rag-badge-dup";
    badge.textContent = t("mcpDuplicateRepeats", { count: repeats });
    // Insert badge as the third grid item so it sits beside the detail line.
    row.appendChild(badge);
  }
  return row;
}

function renderTraceEvent(event) {
  const row = document.createElement("div");
  row.className = `rag-event ${event.type || ""}`;
  if (event.type === "agent_trace" && event.tool) {
    row.classList.add(`tool-${event.tool}`);
  }

  const icon = document.createElement("span");
  icon.className = "rag-event-icon";
  icon.textContent = event.type === "agent_trace"
    ? (TOOL_ICONS[event.tool] || "🛠")
    : (EVENT_ICONS[event.type] || "•");
  const label = document.createElement("strong");
  label.textContent = traceLabel(event);
  const detail = document.createElement("span");
  detail.className = "rag-event-detail";
  detail.textContent = traceDetail(event);
  row.append(icon, label, detail);

  // Inline meta (elapsed, cache_hit) shown as small badges trailing the row.
  const metaBadges = collectTraceMetaBadges(event);
  if (metaBadges.length) {
    const meta = document.createElement("span");
    meta.className = "rag-event-meta";
    for (const badge of metaBadges) {
      const node = document.createElement("span");
      node.className = `rag-badge ${badge.kind || ""}`;
      node.textContent = badge.text;
      meta.appendChild(node);
    }
    row.appendChild(meta);
  }

  if (event.type === "retrieval_done" && Array.isArray(event.groups)) {
    const groupList = document.createElement("div");
    groupList.className = "rag-groups";
    for (const group of event.groups) {
      const item = document.createElement("div");
      item.className = "rag-group";
      const contexts = Array.isArray(group.contexts) ? group.contexts : [];
      item.innerHTML = "<b></b><small></small>";
      item.querySelector("b").textContent = group.source_db
        ? `${group.query || ""} @ ${group.source_db}`
        : group.query || "";
      item.querySelector("small").textContent = contexts
        .map((context) => `${context.entity_name || context.id || "context"} (${context.desc_score || "-"})`)
        .join(" · ");
      groupList.appendChild(item);
    }
    row.appendChild(groupList);
  }

  if (event.type === "agent_trace") {
    const resultNode = renderAgentTraceResult(event);
    if (resultNode) row.appendChild(resultNode);
  }

  return row;
}

function collectTraceMetaBadges(event) {
  const badges = [];
  if (event.type === "agent_trace") {
    const result = event.result || {};
    if (event.tool === "local_file_read" && result.cache_hit) {
      badges.push({ kind: "rag-badge-cache", text: t("mcpCacheHit") });
    }
    if (event.tool === "local_file_read_many" && Array.isArray(result.items)) {
      const cached = result.items.filter((it) => it && it.cache_hit).length;
      if (cached > 0) {
        badges.push({ kind: "rag-badge-cache", text: `${t("mcpCacheHit")} ${cached}/${result.items.length}` });
      }
    }
    if (result && result.ok === false && result.error) {
      badges.push({ kind: "rag-badge-error", text: "✕" });
    }
  }
  return badges;
}

function renderAgentTraceResult(event) {
  const result = event.result || {};
  const tool = event.tool || "";
  const box = document.createElement("div");
  box.className = "rag-groups";

  if (tool === "local_file_read" && result.path) {
    const item = document.createElement("div");
    item.className = "rag-group";
    item.innerHTML = "<b></b><small></small>";
    item.querySelector("b").textContent = compactPath(result.path);
    const flags = [];
    if (result.cache_hit) flags.push(t("mcpCacheHit"));
    if (result.truncated) flags.push("truncated");
    item.querySelector("small").textContent = result.ok
      ? `${result.chars || 0} chars${flags.length ? ` · ${flags.join(" · ")}` : ""}`
      : result.error || "";
    box.appendChild(item);
    return box;
  }

  if (tool === "local_file_read_many" && Array.isArray(result.items)) {
    for (const file of result.items) {
      const item = document.createElement("div");
      item.className = "rag-group";
      item.innerHTML = "<b></b><small></small>";
      const path = file.path || file.request_path || "";
      item.querySelector("b").textContent = compactPath(path);
      const flags = [];
      if (file.cache_hit) flags.push(t("mcpCacheHit"));
      if (file.truncated) flags.push("truncated");
      item.querySelector("small").textContent = file.ok
        ? `${file.chars || 0} chars${flags.length ? ` · ${flags.join(" · ")}` : ""}`
        : file.error || "";
      if (!file.ok) item.classList.add("rag-group-error");
      box.appendChild(item);
    }
    return result.items.length ? box : null;
  }

  if ((tool === "local_file_write" || tool === "local_file_replace") && (result.path || event.arguments?.path)) {
    const item = document.createElement("div");
    item.className = "rag-group";
    item.innerHTML = "<b></b><small></small>";
    item.querySelector("b").textContent = compactPath(result.path || event.arguments?.path || "");
    item.querySelector("small").textContent = result.ok
      ? `${tool === "local_file_write" ? t("mcpWriteFile") : t("mcpReplaceFile")} · ${result.chars || 0} chars`
      : result.error || "";
    if (!result.ok) item.classList.add("rag-group-error");
    box.appendChild(item);
    return box;
  }

  if (tool === "local_file_copy_tree" && (result.target_path || event.arguments?.target_path)) {
    const item = document.createElement("div");
    item.className = "rag-group";
    item.innerHTML = "<b></b><small></small>";
    item.querySelector("b").textContent = result.target_path || event.arguments?.target_path || "";
    item.querySelector("small").textContent = result.ok
      ? `${t("mcpCopyTree")} · ${result.copied_count || 0} files · ${result.created_dir_count || 0} dirs`
      : result.error || "";
    box.appendChild(item);
    return box;
  }

  if (tool === "local_file_create_dir" && (result.path || event.arguments?.path)) {
    const item = document.createElement("div");
    item.className = "rag-group";
    item.innerHTML = "<b></b><small></small>";
    item.querySelector("b").textContent = result.path || event.arguments?.path || "";
    item.querySelector("small").textContent = result.ok
      ? `${t("mcpCreateDir")} · ${result.created ? t("mcpCreated") : t("mcpExists")}`
      : result.error || "";
    box.appendChild(item);
    return box;
  }

  if ((tool === "local_file_search" || tool === "local_file_list") && Array.isArray(result.items)) {
    for (const found of result.items) {
      const item = document.createElement("div");
      item.className = "rag-group";
      item.innerHTML = "<b></b><small></small>";
      item.querySelector("b").textContent = found.path || found.name || "";
      item.querySelector("small").textContent = found.preview || found.type || "";
      box.appendChild(item);
    }
    return result.items.length ? box : null;
  }

  if (tool === "rag_query" && Array.isArray(result.groups)) {
    for (const group of result.groups) {
      const item = document.createElement("div");
      item.className = "rag-group";
      const contexts = Array.isArray(group.contexts) ? group.contexts : [];
      item.innerHTML = "<b></b><small></small>";
      item.querySelector("b").textContent = group.source_db
        ? `${group.query || ""} @ ${group.source_db}`
        : group.query || "";
      item.querySelector("small").textContent = contexts
        .map((context) => {
          const name = context.entity_name || context.id || "context";
          const path = context.mcp_file_path || context.file_path || "";
          return `${name} (${context.desc_score || "-"})${path ? ` @ ${compactPath(path, 36)}` : ""}`;
        })
        .join(" · ");
      box.appendChild(item);
    }
    return result.groups.length ? box : null;
  }

  return null;
}

function traceLabel(event) {
  const labels = {
    start: t("traceStart"),
    memory_loaded: t("traceMemoryLoaded"),
    retrieval_start: t("traceRetrievalStart"),
    retrieval_done: t("traceRetrievalDone"),
    local_files_done: t("traceLocalFiles"),
    prompt_ready: t("tracePromptReady"),
    agent_start: t("traceAgentStart"),
    agent_trace: t("traceAgentTrace"),
    generation_start: t("traceGenerationStart"),
    summary_start: t("traceSummaryStart"),
    memory_updated: t("traceMemoryUpdated"),
    warning: t("traceWarning"),
    done: t("traceDone"),
    aborted: t("traceAborted"),
    error: t("traceError"),
  };
  return labels[event.type] || event.type || "event";
}

function traceDetail(event) {
  if (event.type === "retrieval_start") return event.query || "";
  if (event.type === "retrieval_done") return t("traceContextCount", { count: event.context_count || 0, duration: formatDuration(event.duration_ms) });
  if (event.type === "prompt_ready") return t("tracePromptStats", { count: event.message_count || 0, chars: event.context_chars || 0 });
  if (event.type === "local_files_done") return t("traceToolCount", { count: (event.traces || []).length });
  if (event.type === "agent_trace") return formatAgentTrace(event);
  if (event.type === "agent_start" && event.max_steps) return t("traceAgentBudget", { max: event.max_steps });
  if (event.type === "generation_start") {
    if (event.step === "wrap_up") return t("traceWrapUp");
    if (event.step && event.max_steps) return t("traceStepProgress", { step: event.step, max: event.max_steps });
    return "";
  }
  if (event.type === "memory_updated") return t("traceDuration", { duration: formatDuration(event.duration_ms) });
  if (event.type === "warning" || event.type === "error") return event.message || "";
  if (event.type === "done") return t("traceDoneStats", { duration: formatDuration(event.duration_ms) });
  return event.model || event.summary || event.query || "";
}

function formatAgentTrace(event) {
  const tool = event.tool || "";
  const args = event.arguments || {};
  const result = event.result || {};
  if (tool === "agent_tool_decision") {
    return result.decision || t("traceNoMcpCalls");
  }
  if (tool === "local_file_read") {
    const path = result.path || args.path || "";
    const chars = result.chars ? ` · ${result.chars} chars` : "";
    const cached = result.cache_hit ? ` · ${t("mcpCacheHit")}` : "";
    return `${t("mcpReadFile")}: ${compactPath(path)}${chars}${cached}`;
  }
  if (tool === "local_file_read_many") {
    const paths = Array.isArray(args.paths) ? args.paths : [];
    const okCount = result.ok_count != null ? result.ok_count : (result.items || []).filter((it) => it && it.ok).length;
    const cached = (result.items || []).filter((it) => it && it.cache_hit).length;
    const tail = cached ? ` · ${t("mcpCacheHit")} ${cached}/${paths.length || result.count || 0}` : "";
    return `${t("mcpReadMany")}: ${t("mcpFiles", { count: paths.length || result.count || 0 })} · ok ${okCount}${tail}`;
  }
  if (tool === "local_file_search") {
    const count = Array.isArray(result.items) ? result.items.length : 0;
    const rawPath = result.fallback_path
      ? `${result.path || args.path || "."} → ${result.fallback_path}`
      : result.path || args.path || ".";
    return `${t("mcpSearch")}: ${args.query || ""} @ ${compactPath(rawPath)} · ${count} ${t("mcpResults")}`;
  }
  if (tool === "local_file_list") {
    const count = Array.isArray(result.items) ? result.items.length : 0;
    return `${t("mcpList")}: ${compactPath(result.path || args.path || ".")} · ${count} ${t("mcpResults")}`;
  }
  if (tool === "rag_query") {
    const parts = Array.isArray(result.query_parts) && result.query_parts.length
      ? result.query_parts.join(" / ")
      : result.query || args.query || "";
    const domains = Array.isArray(result.requested_domains) && result.requested_domains.length
      ? ` @ ${result.requested_domains.join(",")}`
      : "";
    return `${t("mcpRagQuery")}: ${parts}${domains} · ${result.context_count || 0} ${t("traceContexts")}`;
  }
  if (tool === "local_file_write") {
    return `${t("mcpWriteFile")}: ${compactPath(result.path || args.path || "")} · ${result.ok ? `${result.chars || 0} chars` : result.error || ""}`;
  }
  if (tool === "local_file_replace") {
    return `${t("mcpReplaceFile")}: ${compactPath(result.path || args.path || "")} · ${result.ok ? `${result.occurrences_replaced || 0}×` : result.error || ""}`;
  }
  if (tool === "local_file_copy_tree") {
    const copied = result.copied_count || 0;
    const dirs = result.created_dir_count || 0;
    const target = compactPath(result.target_path || args.target_path || "");
    return `${t("mcpCopyTree")}: → ${target} · ${result.ok ? `${copied} files · ${dirs} dirs` : result.error || ""}`;
  }
  if (tool === "local_file_create_dir") {
    return `${t("mcpCreateDir")}: ${compactPath(result.path || args.path || "")} · ${result.ok ? (result.created ? t("mcpCreated") : t("mcpExists")) : result.error || ""}`;
  }
  return `${tool} ${JSON.stringify(args)}`;
}

function appendLocalError(content) {
  const conversation = activeConversation();
  if (conversation) {
    conversation.messages.push({ role: "error", content });
    saveState();
  }
  renderActiveConversation();
}

function scrollMessages() {
  // Synchronous: callers run after DOM mutation, inside an rAF or right
  // before paint. Wrapping in another rAF caused a visible flicker frame
  // where the freshly recreated reasoning body briefly sat at scrollTop=0.
  els.messages.scrollTop = els.messages.scrollHeight;
}

function scrollActiveReasoning() {
  const bodies = els.messages.querySelectorAll(".message.assistant.pending .reasoning-body");
  const body = bodies[bodies.length - 1];
  if (body) body.scrollTop = body.scrollHeight;
}

// Original implementation re-built every message node on each
// `reasoning_content`/`token` event (innerHTML="" + appendMessageNode loop),
// which (a) reset the reasoning body's scrollTop to 0 every frame, causing the
// 横跳 flicker between top and bottom, and (b) made it impossible for the user
// to scroll up to read mid-reasoning text — every frame teleported them.
//
// During streaming we now patch the existing pending message node in place:
// only update text content of `.reasoning-body` and `.message-content`. Scroll
// position is preserved unless the user was already at the bottom.
function patchStreamingPending(pending) {
  if (!pending || !pending.id) return false;
  const node = els.messages.querySelector(`[data-message-id="${pending.id}"]`);
  if (!node) return false;
  const bubble = node.querySelector(".bubble");
  if (!bubble) return false;

  if (pending.reasoningContent) {
    let panel = bubble.querySelector(".reasoning-panel");
    let body;
    if (!panel) {
      // First reasoning chunk for this message — build the details once.
      panel = document.createElement("details");
      panel.className = "reasoning-panel";
      panel.open = true;
      const summary = document.createElement("summary");
      summary.textContent = t("reasoningTitle");
      body = document.createElement("div");
      body.className = "reasoning-body";
      panel.append(summary, body);
      bubble.insertBefore(panel, bubble.firstChild);
    } else {
      body = panel.querySelector(".reasoning-body");
    }
    if (body && body.textContent !== pending.reasoningContent) {
      // Auto-stick to bottom only if the user wasn't actively scrolled up.
      const distanceFromBottom = body.scrollHeight - body.scrollTop - body.clientHeight;
      const wasAtBottom = body.scrollTop === 0 || distanceFromBottom < 40;
      body.textContent = pending.reasoningContent;
      if (wasAtBottom) body.scrollTop = body.scrollHeight;
    }
  }

  let content = bubble.querySelector(".message-content");
  if (!content) {
    content = document.createElement("div");
    content.className = "message-content";
    bubble.appendChild(content);
  }
  const nextText = pending.content || "";
  if (content.textContent !== nextText) {
    content.textContent = nextText;
  }
  return true;
}

function isNearMessageBottom() {
  const distance = els.messages.scrollHeight - els.messages.scrollTop - els.messages.clientHeight;
  return distance < 96;
}

function buildUserMessageContent(displayContent, attachments) {
  if (!attachments.length) return displayContent;
  const intro = "The user attached the following files. Use their contents as part of this message.";
  const files = attachments.map((attachment, index) => {
    const extension = ATTACHMENT_EXTENSIONS.has(attachment.extension) ? attachment.extension : "txt";
    const safeContent = String(attachment.content || "").replace(/```/g, "`` `");
    return [
      `Attachment ${index + 1}: ${attachment.name}`,
      `Type: ${attachment.type || extension}`,
      `Size: ${formatBytes(attachment.size)}`,
      "Content:",
      `\`\`\`${extension}`,
      safeContent,
      "```",
    ].join("\n");
  });
  return [displayContent || "Please use the attached file contents.", intro, ...files].filter(Boolean).join("\n\n");
}

async function runCustomDomainFlow(conversation, attachments) {
  const selected = (conversation?.selectedFiles || []).filter(Boolean);
  const attachmentPayload = (attachments || [])
    .filter((item) => item && item.content)
    .map((item) => ({ name: item.name, content: String(item.content || "") }));
  if (!selected.length && !attachmentPayload.length) {
    appendLocalError(t("customDomainNoFiles"));
    return "";
  }

  let response;
  try {
    response = await request("/api/descriptions/split", {
      method: "POST",
      body: JSON.stringify({ selected_files: selected, attachments: attachmentPayload }),
    });
  } catch (error) {
    appendLocalError(t("customDomainFailed", { message: error.message }));
    return "";
  }

  const segments = Array.isArray(response?.segments) ? response.segments : [];
  const domains = Array.isArray(response?.domains) ? response.domains : [];
  if (!segments.length) {
    appendLocalError(t("customDomainEmpty"));
    return "";
  }

  const assignments = await openCustomDomainDialog(segments, domains);
  if (!assignments) return null;

  const groups = new Map();
  for (const seg of segments) {
    const groupId = seg.group_id || `${seg.source}::${seg.key}`;
    if (!groups.has(groupId)) {
      groups.set(groupId, {
        label: seg.group_label || [seg.source, seg.key].filter(Boolean).join(" :: "),
        original: seg.original || "",
        items: [],
      });
    }
    const entry = assignments[seg.id] || {};
    const text = (entry.text ?? seg.text).trim();
    if (!text) continue;
    const ids = Array.isArray(entry.domains) ? entry.domains : (entry.domain ? [entry.domain] : []);
    const labels = ids.map((id) => domains.find((item) => item.id === id)?.label || id);
    const labelStr = labels.length ? labels.join(", ") : "(unset)";
    groups.get(groupId).items.push({ label: labelStr, text });
  }

  const lines = [`### ${t("customDomainSection")}`];
  let groupIndex = 0;
  for (const group of groups.values()) {
    if (!group.items.length) continue;
    groupIndex += 1;
    const tag = `G${groupIndex}`;
    lines.push("");
    lines.push(`#### [${tag}] ${group.label}`);
    if (group.original) lines.push(`> ${group.original.replace(/\n+/g, " ")}`);
    for (const item of group.items) {
      lines.push(`- [${tag}] [${item.label}] ${item.text}`);
    }
  }
  return lines.length > 1 ? lines.join("\n") : "";
}

function openCustomDomainDialog(segments, domains) {
  return new Promise((resolve) => {
    const dialog = els.customDomainDialog;
    const list = els.customDomainList;
    if (!dialog || !list) {
      resolve(null);
      return;
    }
    list.innerHTML = "";
    const rows = new Map();
    const defaultDomain = domains.some((item) => item.id === "cards")
      ? "cards"
      : (domains[0]?.id || "");

    const groupOrder = [];
    const grouped = new Map();
    for (const seg of segments) {
      const groupId = seg.group_id || `${seg.source}::${seg.key}`;
      if (!grouped.has(groupId)) {
        grouped.set(groupId, {
          label: seg.group_label || [seg.source, seg.key].filter(Boolean).join(" :: "),
          original: seg.original || "",
          items: [],
        });
        groupOrder.push(groupId);
      }
      grouped.get(groupId).items.push(seg);
    }

    let groupIndex = 0;
    for (const groupId of groupOrder) {
      groupIndex += 1;
      const group = grouped.get(groupId);
      const block = document.createElement("section");
      block.className = "custom-domain-group";

      const header = document.createElement("header");
      header.className = "custom-domain-group-header";
      const tag = document.createElement("span");
      tag.className = "custom-domain-group-tag";
      tag.textContent = `G${groupIndex}`;
      const label = document.createElement("span");
      label.className = "custom-domain-group-label";
      label.textContent = group.label;
      header.appendChild(tag);
      header.appendChild(label);
      block.appendChild(header);

      if (group.original) {
        const original = document.createElement("p");
        original.className = "custom-domain-group-original";
        original.textContent = group.original;
        block.appendChild(original);
      }

      for (const seg of group.items) {
        const row = document.createElement("div");
        row.className = "custom-domain-row";

        const textWrap = document.createElement("div");
        const text = document.createElement("textarea");
        text.className = "segment-text";
        text.rows = 2;
        text.value = seg.text;
        textWrap.appendChild(text);

        const domainBox = document.createElement("div");
        domainBox.className = "custom-domain-checkboxes";
        const checkboxes = [];
        for (const domain of domains) {
          const opt = document.createElement("label");
          opt.className = "custom-domain-checkbox";
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.value = domain.id;
          if (domain.id === defaultDomain) cb.checked = true;
          const lbl = document.createElement("span");
          lbl.textContent = domain.label;
          opt.appendChild(cb);
          opt.appendChild(lbl);
          domainBox.appendChild(opt);
          checkboxes.push(cb);
        }
        rows.set(seg.id, { checkboxes, text });

        row.appendChild(textWrap);
        row.appendChild(domainBox);
        block.appendChild(row);
      }
      list.appendChild(block);
    }

    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      els.customDomainConfirm.removeEventListener("click", onConfirm);
      dialog.removeEventListener("close", onClose);
      resolve(value);
    };
    const onConfirm = () => {
      const result = {};
      for (const [id, entry] of rows.entries()) {
        const domains = entry.checkboxes.filter((cb) => cb.checked).map((cb) => cb.value);
        result[id] = { domains, text: entry.text.value };
      }
      if (dialog.open) dialog.close("confirm");
      finish(result);
    };
    const onClose = () => {
      if (dialog.returnValue !== "confirm") finish(null);
    };

    els.customDomainConfirm.addEventListener("click", onConfirm);
    dialog.addEventListener("close", onClose);
    dialog.returnValue = "";
    dialog.showModal();
  });
}

async function sendMessage() {
  const conversation = activeConversation();
  let displayContent = els.messageInput.value.trim();
  const attachments = (conversation?.draftAttachments || []).slice();
  if (!conversation || (!displayContent && !attachments.length) || state.sending) return;

  if (els.customDomainToggle?.checked) {
    const enriched = await runCustomDomainFlow(conversation, attachments);
    if (enriched === null) return; // user canceled
    if (enriched) {
      const current = els.messageInput.value;
      els.messageInput.value = current ? `${current}\n\n${enriched}` : enriched;
      displayContent = els.messageInput.value.trim();
      resizeComposer();
    }
  }
  const content = buildUserMessageContent(displayContent, attachments);
  const attachmentMeta = attachments.map(({ id, name, extension, type, size, lastModified }) => ({
    id,
    name,
    extension,
    type,
    size,
    lastModified,
  }));

  conversation.messages.push({ role: "user", content, displayContent, attachments: attachmentMeta });
  if (conversation.autoTitle) {
    conversation.title = (displayContent || attachmentMeta.map((item) => item.name).join(", ")).slice(0, 24);
    conversation.autoTitle = false;
  }
  conversation.updatedAt = new Date().toISOString();
  els.messageInput.value = "";
  conversation.draftAttachments = [];
  resizeComposer();
  renderComposerAttachments();
  state.sending = true;
  state.sendingId = conversation.id;
  state.abortController = new AbortController();
  render();

  const pending = { id: uid(), role: "assistant", content: "", reasoningContent: "", pending: true, events: [] };
  conversation.messages.push(pending);
  renderActiveConversation();
  scrollMessages();
  updateSendButton();

  try {
    await streamChat(conversation, pending);
    finalizeStreamWithoutDone(conversation, pending);
  } catch (error) {
    if (error.name === "AbortError") {
      pending.events.push({ type: "aborted", message: t("aborted") });
      pending.content ||= t("aborted");
    } else {
      pending.role = "error";
      pending.content = error.message;
      pending.events.push({ type: "error", message: error.message });
    }
    pending.pending = false;
  } finally {
    state.sending = false;
    state.sendingId = "";
    state.abortController = null;
    conversation.updatedAt = new Date().toISOString();
    flushSaveState();
    render();
  }
}

async function streamChat(conversation, pending) {
  const response = await fetch(`${API}/api/chat/stream`, {
    method: "POST",
    signal: state.abortController.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversation.id,
      messages: conversation.messages
        .filter((item) => item.role === "user" || item.role === "assistant")
        .slice(0, -1),
      provider_id: conversation.providerId,
      model: conversation.model || els.modelInput.value.trim(),
      use_rag: els.ragToggle.checked,
      use_agent: els.agentToggle.checked,
      context_n: Number(els.contextN.value || 3),
      desc_top_k: Number(els.descTopK.value || 4),
      selected_files: conversation.selectedFiles,
      memory_summary: conversation.memorySummary || "",
      language: state.lang,
    }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  if (!response.body) {
    throw new Error("Streaming response body is not available.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch (err) {
        console.warn("Skipping malformed SSE line", { error: err && err.message, sample: line.slice(0, 200), bytes: line.length });
        continue;
      }
      try {
        handleStreamEvent(conversation, pending, event);
      } catch (err) {
        console.error("handleStreamEvent threw", err, event && event.type);
      }
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    try {
      handleStreamEvent(conversation, pending, JSON.parse(buffer));
    } catch (err) {
      console.warn("Skipping trailing malformed SSE buffer", err && err.message);
    }
  }
}

function collectCompletionFromEvents(events) {
  const stats = summarizeRagEvents(events);
  const writeTools = new Set([
    "local_file_write",
    "local_file_replace",
    "local_file_copy_tree",
    "local_file_create_dir",
  ]);
  const files = [];
  for (const event of events) {
    if (!event || event.type !== "agent_trace") continue;
    const tool = event.tool || "";
    const result = event.result || {};
    const args = event.arguments || {};
    if (!writeTools.has(tool) || !result.ok) continue;
    const path = result.path || result.target_path || args.path || args.target_path || "";
    if (path && !files.includes(path)) files.push(path);
  }

  let durationMs = 0;
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i] || {};
    durationMs = event.duration_ms || event.elapsed_ms || 0;
    if (durationMs) break;
  }
  if (!durationMs) {
    const timestamps = events.map((event) => Number(event && event.ts)).filter((ts) => Number.isFinite(ts));
    if (timestamps.length >= 2) {
      durationMs = Math.max(0, timestamps[timestamps.length - 1] - timestamps[0]);
    }
  }

  return {
    status: "ok",
    writes: stats.local_file_write + stats.local_file_replace + stats.local_file_copy_tree + stats.local_file_create_dir,
    files: files.slice(0, 12),
    duration_ms: durationMs,
    use_agent: events.some((event) => event && event.type === "agent_start"),
  };
}

function finalizeStreamWithoutDone(conversation, pending) {
  if (!pending || !pending.pending || pending.role !== "assistant") return;
  if (!String(pending.content || "").trim()) return;
  const events = Array.isArray(pending.events) ? pending.events : [];
  if (events.some((event) => event && ["done", "error", "aborted"].includes(event.type))) return;

  const completion = collectCompletionFromEvents(events);
  pending.pending = false;
  pending.meta = t("assistantMeta", {
    provider: conversation.providerId || "",
    model: conversation.model || els.modelInput.value.trim() || "",
    context: 0,
    tools: summarizeRagEvents(events).tool_total,
  });
  pending.completion = completion;
  pending.events = [
    ...events,
    {
      type: "done",
      inferred: true,
      duration_ms: completion.duration_ms,
      completion: "ok",
      writes_count: completion.writes,
      written_files: completion.files,
      use_agent: completion.use_agent,
    },
  ];
}

let _streamRenderHandle = null;
let _streamRenderStick = false;
function scheduleStreamingRender(stickToBottom /*, scrollReasoning */) {
  if (stickToBottom) _streamRenderStick = true;
  if (_streamRenderHandle) return;
  _streamRenderHandle = requestAnimationFrame(() => {
    _streamRenderHandle = null;
    const stick = _streamRenderStick;
    _streamRenderStick = false;
    const conversation = activeConversation();
    const pending = conversation?.messages?.find((message) => message.pending);
    const patched = pending ? patchStreamingPending(pending) : false;
    if (!patched) {
      // Fallback: pending node not in DOM yet (first render of a new message).
      renderActiveConversation();
      renderConversationList();
      updateSendButton();
    }
    if (stick) scrollMessages();
  });
}

let _saveStateHandle = null;
function scheduleSaveState() {
  if (_saveStateHandle) return;
  _saveStateHandle = setTimeout(() => {
    _saveStateHandle = null;
    saveState();
  }, 600);
}
function flushSaveState() {
  if (_saveStateHandle) {
    clearTimeout(_saveStateHandle);
    _saveStateHandle = null;
  }
  saveState();
}

function handleStreamEvent(conversation, pending, event) {
  const stickToBottom = isNearMessageBottom();
  const isStreamingToken = event.type === "token" || event.type === "reasoning_content";
  if (event.type === "token") {
    pending.content += event.text || "";
  } else if (event.type === "reasoning_content") {
    pending.reasoningContent = `${pending.reasoningContent || ""}${event.text || ""}`;
  } else if (event.type === "done") {
    pending.pending = false;
    if (event.answer && !pending.content) pending.content = event.answer;
    if (event.reasoning_content) pending.reasoningContent = event.reasoning_content;
    pending.meta = t("assistantMeta", {
      provider: event.provider_id || "",
      model: event.model || "",
      context: event.context_count || 0,
      tools: event.agent_trace_count || 0,
    });
    pending.completion = {
      status: event.completion || "ok",
      writes: event.writes_count || 0,
      files: Array.isArray(event.written_files) ? event.written_files : [],
      duration_ms: event.duration_ms || 0,
      use_agent: !!event.use_agent,
    };
    conversation.memorySummary = event.memory_summary || conversation.memorySummary || "";
    pending.memorySummary = conversation.memorySummary;
  } else if (event.type === "memory_updated") {
    conversation.memorySummary = event.summary || conversation.memorySummary || "";
    pending.memorySummary = conversation.memorySummary;
  } else if (event.type === "error") {
    pending.role = "error";
    pending.content = event.message || t("streamError");
    pending.pending = false;
  }

  if (event.reasoning_content) delete event.reasoning_content;
  if (event.type !== "token") {
    pending.events ||= [];
    if (event.type !== "reasoning_content") pending.events.push(event);
  }
  conversation.updatedAt = new Date().toISOString();

  // The "done" event flips pending=false and adds completion metadata. The
  // streaming patch path doesn't render new bubble children, so fall through
  // to the full-render branch below to surface the completion badge.
  if (isStreamingToken && pending.pending) {
    scheduleSaveState();
    scheduleStreamingRender(stickToBottom);
    return;
  }

  flushSaveState();
  renderActiveConversation();
  if (stickToBottom) scrollMessages();
  renderConversationList();
  updateSendButton();
}

function resizeComposer() {
  els.messageInput.style.height = "auto";
  els.messageInput.style.height = `${Math.min(els.messageInput.scrollHeight, 180)}px`;
}

function renderSettings() {
  if (!state.config) {
    els.providerSettings.textContent = t("settingsNotLoaded");
    return;
  }
  els.providerSettings.innerHTML = "";
  for (const provider of state.config.catalog) {
    const saved = state.config.providers?.[provider.id] || {};
    const card = document.createElement("section");
    card.className = "provider-card";
    card.innerHTML = `
      <div class="provider-name">
        <strong></strong>
        <span class="status-line"></span>
      </div>
      <label><span class="label-api-key"></span><input class="provider-key" type="password" autocomplete="off" /></label>
      <label><span class="label-base-url"></span><input class="provider-url" /></label>
      <label><span class="label-model"></span><input class="provider-model" list="models-${provider.id}" /><datalist id="models-${provider.id}"></datalist></label>
      <label><span class="label-context-length"></span><input class="provider-context-length" type="number" min="0" step="1024" /></label>
      <div class="provider-actions">
        <button class="provider-save" type="button"></button>
        <button class="provider-clear" type="button"></button>
      </div>
    `;
    card.querySelector("strong").textContent = provider.name;
    card.querySelector(".status-line").textContent = saved.has_key ? t("configured", { source: saved.key_source }) : t("unconfigured").trim();
    card.querySelector(".label-api-key").textContent = t("labelApiKey");
    card.querySelector(".label-base-url").textContent = t("labelBaseUrl");
    card.querySelector(".label-model").textContent = t("labelModel");
    card.querySelector(".label-context-length").textContent = t("labelContextLength");
    card.querySelector(".provider-key").placeholder = t("apiKeyPlaceholder");
    card.querySelector(".provider-url").value = saved.base_url || provider.base_url || "";
    card.querySelector(".provider-model").value = saved.model || provider.models?.[0] || "";
    const contextLengthInput = card.querySelector(".provider-context-length");
    contextLengthInput.value = Number.isFinite(saved.context_length) ? saved.context_length : 262144;
    contextLengthInput.placeholder = t("contextLengthPlaceholder");
    card.querySelector(".provider-save").textContent = t("save");
    card.querySelector(".provider-clear").textContent = t("clearKey");
    const datalist = card.querySelector("datalist");
    for (const model of provider.models || []) {
      const option = document.createElement("option");
      option.value = model;
      datalist.appendChild(option);
    }
    card.querySelector(".provider-save").addEventListener("click", () => saveProvider(provider.id, card, false));
    card.querySelector(".provider-clear").addEventListener("click", () => saveProvider(provider.id, card, true));
    els.providerSettings.appendChild(card);
  }
}

async function saveProvider(providerId, card, clearKey) {
  const saveButton = card.querySelector(".provider-save");
  saveButton.textContent = t("saving");
  try {
    const contextLengthRaw = card.querySelector(".provider-context-length").value.trim();
    const contextLengthValue = contextLengthRaw === "" ? 262144 : Math.max(0, parseInt(contextLengthRaw, 10) || 0);
    state.config = await request(`/api/config/providers/${providerId}`, {
      method: "POST",
      body: JSON.stringify({
        api_key: card.querySelector(".provider-key").value.trim(),
        base_url: card.querySelector(".provider-url").value.trim(),
        model: card.querySelector(".provider-model").value.trim(),
        context_length: contextLengthValue,
        default_provider_id: providerId,
        clear_key: clearKey,
      }),
    });
    const conversation = activeConversation();
    if (conversation) {
      conversation.providerId = providerId;
      conversation.model = state.config.providers?.[providerId]?.model || conversation.model;
    }
    saveButton.textContent = t("saved");
    saveState();
    renderProviderSelect();
    renderMeta();
    setTimeout(renderSettings, 350);
  } catch (error) {
    saveButton.textContent = error.message;
  }
}

async function saveDefaultProvider() {
  if (!els.providerSelect.value) return;
  try {
    state.config = await request("/api/config/default", {
      method: "POST",
      body: JSON.stringify({ provider_id: els.providerSelect.value }),
    });
    renderProviderSelect();
  } catch (error) {
    appendLocalError(error.message);
  }
}

async function listFiles(path) {
  els.fileBrowser.textContent = t("loading");
  try {
    const data = await request(`/api/mcp/files/list?path=${encodeURIComponent(path)}`, {
      method: "GET",
    });
    if (!data.ok) throw new Error(data.error || t("listFailed"));
    els.filePathInput.value = data.path || ".";
    els.fileBrowser.innerHTML = "";
    if (data.path && data.path !== ".") {
      appendFileRow({ type: "directory", path: parentPath(data.path), name: ".." });
    }
    for (const item of data.items) {
      appendFileRow(item);
    }
  } catch (error) {
    els.fileBrowser.textContent = error.message;
  }
}

function appendFileRow(item) {
  const row = document.createElement("div");
  row.className = "file-row";
  row.innerHTML = "<span class=\"file-type\"></span><span></span><button class=\"file-add\" type=\"button\"></button>";
  row.querySelector(".file-type").textContent = item.type === "directory" ? t("fileTypeDirectory") : t("fileTypeFile");
  row.querySelector("span:nth-child(2)").textContent = item.path;
  const button = row.querySelector("button");
  if (item.type === "directory") {
    button.textContent = t("open");
    button.addEventListener("click", () => listFiles(item.path));
    row.addEventListener("dblclick", () => listFiles(item.path));
  } else {
    button.textContent = t("add");
    button.addEventListener("click", () => {
      const conversation = activeConversation();
      if (!conversation) return;
      conversation.selectedFiles ||= [];
      if (!conversation.selectedFiles.includes(item.path)) {
        conversation.selectedFiles.push(item.path);
      }
      saveState();
      renderFileChips();
    });
  }
  els.fileBrowser.appendChild(row);
}

function parentPath(path) {
  const parts = String(path).split("/").filter(Boolean);
  parts.pop();
  return parts.join("/") || ".";
}

function formatTime(value) {
  try {
    return new Date(value).toLocaleString(state.lang === "zh" ? "zh-CN" : "en-US", { hour12: false });
  } catch {
    return "";
  }
}

boot();
