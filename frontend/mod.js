'use strict';

// ── i18n helper ───────────────────────────────────────
function t(key, params = {}) {
  return window.STS2_SHARED ? window.STS2_SHARED.t(key, params) : key;
}

// ── State ─────────────────────────────────────────────
let currentMod  = null;   // null = root view, string = inside a mod
let currentData = null;   // cached data for re-render on lang change

// ── DOM refs ──────────────────────────────────────────
const pathBar        = document.getElementById('pathBar');
const explorerContent = document.getElementById('explorerContent');

// ── Bootstrap ─────────────────────────────────────────
// shared.js fires DOMContentLoaded first; mod.js runs after it, so
// we wait for the same event (or just call directly if already loaded).
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadMods);
} else {
  loadMods();
}

// Re-render on language switch (fired by shared.js)
window.addEventListener('sts2langchange', () => {
  if (currentMod === null) {
    if (currentData) renderFolders(currentData);
  } else {
    if (currentData) renderFiles(currentData, currentMod);
  }
  setPath(currentMod ? [currentMod] : []);
});

// ── Data fetching ─────────────────────────────────────
async function loadMods() {
  currentMod  = null;
  currentData = null;
  showLoading();
  setPath([]);
  try {
    const res  = await fetch('/api/mods');
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Failed to load mods');
    currentData = data.mods || [];
    renderFolders(currentData);
  } catch (err) {
    showError(err.message);
  }
}

async function loadModFiles(modName) {
  currentMod  = modName;
  currentData = null;
  showLoading();
  setPath([modName]);
  try {
    const res  = await fetch(`/api/mods/${encodeURIComponent(modName)}/models`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Failed to load files');
    currentData = data.files || [];
    renderFiles(currentData, modName);
  } catch (err) {
    showError(err.message);
  }
}

// ── Rendering ─────────────────────────────────────────
function renderFolders(mods) {
  if (!mods.length) {
    explorerContent.innerHTML = `<div class="explorer-empty">${esc(t('modEmpty'))}</div>`;
    return;
  }

  const bar = el('div', 'section-bar');
  bar.innerHTML = `<h2>${esc(t('modSectionProjects'))}</h2><span>${esc(t('modFolderCount', { count: mods.length }))}</span>`;

  const grid = el('div', 'folder-grid');
  for (const mod of mods) {
    const item = el('div', 'folder-item');
    item.setAttribute('role', 'button');
    item.setAttribute('tabindex', '0');
    item.innerHTML = `
      <span class="folder-icon">📁</span>
      <span class="folder-name">${esc(mod.name)}</span>
    `;
    item.addEventListener('click', () => loadModFiles(mod.name));
    item.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') loadModFiles(mod.name);
    });

    const delBtn = el('button', 'folder-delete-btn');
    delBtn.type = 'button';
    delBtn.title = t('modDeleteTip');
    delBtn.setAttribute('aria-label', t('modDeleteTip'));
    delBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
        <path d="M10 11v6"></path>
        <path d="M14 11v6"></path>
        <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"></path>
      </svg>
    `;
    delBtn.addEventListener('click', e => {
      e.stopPropagation();
      deleteMod(mod.name, item);
    });
    delBtn.addEventListener('keydown', e => {
      // Prevent the parent folder-item key handler from also firing.
      if (e.key === 'Enter' || e.key === ' ') e.stopPropagation();
    });
    item.appendChild(delBtn);

    grid.appendChild(item);
  }

  explorerContent.innerHTML = '';
  explorerContent.appendChild(bar);
  explorerContent.appendChild(grid);
}

// Strip the [color]...[/color] markup used in localization strings; preserve
// the inner text. Used for things like img alt where rich rendering is N/A.
function stripLocMarkup(text) {
  if (!text) return '';
  let s = String(text);
  s = s.replace(/\[\/?\w+\]/g, '');
  s = s.replace(/\\n/g, ' ').replace(/\n/g, ' ');
  return s;
}

// Tags recognized inside localization strings. Unknown tags are dropped
// (wrapper removed) but their inner text is preserved.
const LOC_KNOWN_TAGS = new Set([
  'aqua', 'blue', 'gold', 'green', 'orange', 'pink', 'purple', 'red', 'rainbow',
  'b', 'i', 'center',
  'jitter', 'shake', 'sine', 'thinky_dots',
]);

// Render the raw localization string as HTML: literal "\n" and real newlines
// become <br>, and [tag]…[/tag] (including nested) becomes a styled <span>.
function renderLocMarkupHTML(raw) {
  if (raw === null || raw === undefined) return '';
  let s = esc(String(raw));
  // Resolve innermost [tag]...[/tag] first; repeat to unwind nesting.
  const innermost = /\[(\w+)\]([^\[\]]*)\[\/\1\]/;
  for (let i = 0; i < 16; i++) {
    let hit = false;
    s = s.replace(innermost, (_m, tag, inner) => {
      hit = true;
      if (!LOC_KNOWN_TAGS.has(tag)) return inner;
      return `<span class="loc-tag loc-tag-${tag}">${inner}</span>`;
    });
    if (!hit) break;
  }
  s = s.replace(/\\n/g, '<br>').replace(/\n/g, '<br>');
  return s;
}

// Inverse of "edit shows raw text": when the user commits an edit, real
// newlines they typed (Shift+Enter) become the literal \n the JSON stores.
function normalizeEditedText(text) {
  return String(text ?? '').replace(/\r\n?/g, '\n').replace(/\n/g, '\\n');
}

function renderFiles(files, modName) {
  if (!files.length) {
    explorerContent.innerHTML = `<div class="explorer-empty">${esc(t('modNoFiles'))}</div>`;
    return;
  }

  const groups = new Map();
  for (const file of files) {
    const cat = file.category || '_root';
    if (!groups.has(cat)) groups.set(cat, []);
    groups.get(cat).push(file);
  }
  const cats = [...groups.keys()].sort();

  const bar = el('div', 'section-bar');
  const titleWrap = el('div', 'section-bar-title');
  titleWrap.innerHTML = `<h2>${esc(modName)}</h2><span>${esc(t('modFileCount', { count: files.length }))}</span>`;
  bar.appendChild(titleWrap);

  const exportBtn = el('button', 'mod-export-btn');
  exportBtn.type = 'button';
  exportBtn.textContent = t('modExportBtn');
  exportBtn.addEventListener('click', () => exportMod(modName, exportBtn));
  bar.appendChild(exportBtn);

  const board = el('div', 'category-board');

  for (const cat of cats) {
    const col = el('div', 'category-col');
    const head = el('div', 'category-head');
    head.textContent = cat === '_root' ? '/' : cat;
    col.appendChild(head);

    for (const file of groups.get(cat)) {
      const wrap = el('div', 'category-entry');
      const row = el('div', 'category-item');
      const expected = (file.assets_expected || []).length;
      const found = (file.assets_found || []).length;
      const missing = (file.assets_missing || []);
      let status = 'none', icon = '·';
      if (expected > 0) {
        if (missing.length === 0) { status = 'ok'; icon = '✓'; }
        else if (found > 0) { status = 'partial'; icon = '!'; }
        else { status = 'missing'; icon = '✗'; }
      }
      const tipLines = [file.rel_path, `snake: ${file.snake_name || ''}`];
      if (file.assets_found && file.assets_found.length) {
        tipLines.push('found:');
        for (const a of file.assets_found) tipLines.push(`  ${a.key}: ${a.path}`);
      }
      row.title = tipLines.join('\n');
      row.innerHTML = `
        <span class="file-row-icon">📄</span>
        <span class="file-row-name">${esc(file.class_name || file.name)}</span>
        <span class="asset-badge asset-${status}">${esc(icon)}</span>
        <span class="asset-detail">${esc(file.snake_name || '')}</span>
      `;
      wrap.appendChild(row);

      const editableCategory = file.category && /^[A-Za-z0-9_-]+$/.test(file.category);
      const editableSnake = file.snake_name && /^[a-z0-9_]+$/.test(file.snake_name);
      const canEdit = !!(editableCategory && editableSnake);

      if (canEdit || file.image_path || (file.localization && Object.keys(file.localization).length)) {
        const lang = window.STS2_SHARED ? window.STS2_SHARED.getLang() : 'zh';
        const loc = (file.localization && (file.localization[lang] || file.localization.en || file.localization.zh)) || null;
        const titleRaw = loc ? String(loc.title ?? '') : '';
        const descRaw  = loc ? String(loc.description ?? '') : '';

        if (file.image_path || titleRaw || descRaw || canEdit) {
          const card = el('div', 'loc-card');
          if (file.image_path) {
            const img = el('img', 'loc-card-img');
            img.alt = stripLocMarkup(titleRaw) || file.snake_name || '';
            img.loading = 'lazy';
            img.src = `/api/mods/${encodeURIComponent(modName)}/file?path=${encodeURIComponent(file.image_path)}`;
            card.appendChild(img);
          }
          const body = el('div', 'loc-card-body');
          const h = el('div', 'loc-card-title');
          h.innerHTML = renderLocMarkupHTML(titleRaw);
          if (canEdit) bindLocEdit(h, modName, file, lang, 'title');
          body.appendChild(h);

          const d = el('div', 'loc-card-desc');
          d.innerHTML = renderLocMarkupHTML(descRaw);
          if (canEdit) bindLocEdit(d, modName, file, lang, 'description');
          body.appendChild(d);

          card.appendChild(body);
          wrap.appendChild(card);
        }
      }

      if (missing.length) {
        const miss = el('div', 'category-missing');
        const label = el('span', 'category-missing-label');
        label.textContent = t('modMissingLabel');
        miss.appendChild(label);
        for (const m of missing) {
          const link = el('button', 'missing-link');
          link.type = 'button';
          link.textContent = m.path;
          link.title = t('modMissingClickTip');
          link.addEventListener('click', () => openFolder(modName, m.path));
          miss.appendChild(link);
        }
        wrap.appendChild(miss);
      }

      col.appendChild(wrap);
    }
    board.appendChild(col);
  }

  explorerContent.innerHTML = '';
  explorerContent.appendChild(bar);
  explorerContent.appendChild(board);
}

function bindLocEdit(node, modName, file, lang, field) {
  node.classList.add('loc-editable');
  if (!node.innerHTML) node.classList.add('loc-empty');
  node.title = t('modLocEditTip');

  let editing = false;
  let originalRaw = (() => {
    const loc = (file.localization && (file.localization[lang] || file.localization.en || file.localization.zh)) || null;
    return loc ? String(loc[field] ?? '') : '';
  })();

  const renderDisplay = (raw) => {
    node.innerHTML = renderLocMarkupHTML(raw);
    node.classList.toggle('loc-empty', !raw);
  };

  const beginEdit = () => {
    if (editing) return;
    editing = true;
    node.contentEditable = 'plaintext-only';
    if (node.contentEditable !== 'plaintext-only') node.contentEditable = 'true';
    node.classList.add('loc-editing');
    node.classList.remove('loc-empty');
    // Show the raw source: literal "\n", [color] tags and all.
    node.textContent = originalRaw;
    node.focus();
    const sel = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(node);
    range.collapse(false);
    sel.removeAllRanges();
    sel.addRange(range);
  };

  const cancelEdit = () => {
    if (!editing) return;
    editing = false;
    node.contentEditable = 'false';
    node.classList.remove('loc-editing');
    renderDisplay(originalRaw);
  };

  const commitEdit = async () => {
    if (!editing) return;
    editing = false;
    node.contentEditable = 'false';
    node.classList.remove('loc-editing');
    // textContent may contain real \n from Shift+Enter — normalize them to
    // the literal "\n" the JSON files use, so round-trips stay consistent.
    const newText = normalizeEditedText(node.textContent).replace(/\s+$/g, '');
    if (newText === originalRaw) {
      renderDisplay(originalRaw);
      return;
    }
    node.classList.add('loc-saving');
    node.textContent = newText;
    try {
      await saveLocalization(modName, file, lang, field, newText);
      if (!file.localization) file.localization = {};
      if (!file.localization[lang]) file.localization[lang] = { title: '', description: '' };
      file.localization[lang][field] = newText;
      originalRaw = newText;
      renderDisplay(originalRaw);
    } catch (err) {
      renderDisplay(originalRaw);
      showError(t('modLocSaveFailed', { message: err.message || String(err) }));
    } finally {
      node.classList.remove('loc-saving');
    }
  };

  node.addEventListener('click', () => { if (!editing) beginEdit(); });
  node.addEventListener('keydown', (e) => {
    if (!editing) return;
    if (e.key === 'Escape') { e.preventDefault(); cancelEdit(); }
    else if (e.key === 'Enter' && field === 'title') { e.preventDefault(); commitEdit(); }
    else if (e.key === 'Enter' && !e.shiftKey && field === 'description') {
      // Description: bare Enter commits; Shift+Enter inserts a newline that
      // commitEdit will store as the literal "\n".
      e.preventDefault();
      commitEdit();
    }
  });
  node.addEventListener('blur', () => { if (editing) commitEdit(); });
}

async function saveLocalization(modName, file, lang, field, value) {
  const res = await fetch(`/api/mods/${encodeURIComponent(modName)}/localization`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      lang,
      category: file.category,
      snake: file.snake_name,
      field,
      value,
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || ('HTTP ' + res.status));
  }
  return data;
}

async function deleteMod(modName, itemEl) {
  const message = t('modDeleteConfirm', { name: modName });
  if (!window.confirm(message)) return;

  if (itemEl) {
    itemEl.classList.add('folder-item-deleting');
    itemEl.style.pointerEvents = 'none';
  }
  try {
    const res = await fetch(`/api/mods/${encodeURIComponent(modName)}`, { method: 'DELETE' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || ('HTTP ' + res.status));
    }
    await loadMods();
  } catch (err) {
    if (itemEl) {
      itemEl.classList.remove('folder-item-deleting');
      itemEl.style.pointerEvents = '';
    }
    showError(t('modDeleteFailed', { message: err.message || String(err) }));
  }
}

async function exportMod(modName, btn) {
  if (!btn) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = t('modExporting');
  try {
    const res = await fetch(`/api/mods/${encodeURIComponent(modName)}/export`, {
      method: 'POST',
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || ('HTTP ' + res.status));
    }
    btn.textContent = t('modExportStarted');
    setTimeout(() => {
      btn.textContent = original;
      btn.disabled = false;
    }, 1800);
  } catch (err) {
    btn.textContent = original;
    btn.disabled = false;
    showError(t('modExportFailed', { message: err.message || String(err) }));
  }
}

async function openFolder(modName, relPath) {
  try {
    const res = await fetch(`/api/mods/${encodeURIComponent(modName)}/open-folder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: relPath }),
    });
    const data = await res.json();
    if (!data.ok) showError(data.error || 'open failed');
  } catch (err) {
    showError(err.message);
  }
}

// ── Path breadcrumb ───────────────────────────────────
function setPath(segments) {
  pathBar.innerHTML = '';

  const homeBtn = el('button', 'path-home');
  homeBtn.title = t('modPathHome');
  homeBtn.textContent = '🏠';
  homeBtn.addEventListener('click', loadMods);
  pathBar.appendChild(homeBtn);

  const rootSeg = el('button', 'path-segment' + (segments.length === 0 ? ' current' : ''));
  rootSeg.textContent = 'mods';
  if (segments.length > 0) rootSeg.addEventListener('click', loadMods);
  pathBar.appendChild(sep());
  pathBar.appendChild(rootSeg);

  for (let i = 0; i < segments.length; i++) {
    pathBar.appendChild(sep());
    const seg = el('button', 'path-segment' + (i === segments.length - 1 ? ' current' : ''));
    seg.textContent = segments[i];
    if (i === 0 && segments.length > 1) {
      seg.addEventListener('click', () => loadModFiles(segments[0]));
    }
    pathBar.appendChild(seg);
  }
}

// ── UI helpers ────────────────────────────────────────
function showLoading() {
  explorerContent.innerHTML = `<div class="explorer-loading">${esc(t('modLoading'))}</div>`;
}

function showError(msg) {
  explorerContent.innerHTML = `<div class="explorer-error">⚠ ${esc(msg)}</div>`;
}

function sep() {
  const s = el('span', 'path-separator');
  s.textContent = '›';
  return s;
}

function el(tag, className) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  return e;
}

function esc(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
