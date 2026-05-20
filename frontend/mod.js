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
// the inner text. Also turn smartDescription placeholders like {Amount} into
// a readable form.
function stripLocMarkup(text) {
  if (!text) return '';
  let s = String(text);
  s = s.replace(/\[\/?\w+\]/g, '');
  return s;
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

      if (file.image_path || (file.localization && Object.keys(file.localization).length)) {
        const lang = window.STS2_SHARED ? window.STS2_SHARED.getLang() : 'zh';
        const loc = (file.localization && (file.localization[lang] || file.localization.en || file.localization.zh)) || null;
        const title = loc ? stripLocMarkup(loc.title) : '';
        const desc  = loc ? stripLocMarkup(loc.description) : '';

        if (file.image_path || title || desc) {
          const card = el('div', 'loc-card');
          if (file.image_path) {
            const img = el('img', 'loc-card-img');
            img.alt = title || file.snake_name || '';
            img.loading = 'lazy';
            img.src = `/api/mods/${encodeURIComponent(modName)}/file?path=${encodeURIComponent(file.image_path)}`;
            card.appendChild(img);
          }
          const body = el('div', 'loc-card-body');
          if (title) {
            const h = el('div', 'loc-card-title');
            h.textContent = title;
            body.appendChild(h);
          }
          if (desc) {
            const d = el('div', 'loc-card-desc');
            d.textContent = desc;
            body.appendChild(d);
          }
          card.appendChild(body);
          wrap.appendChild(card);
        }
      }

      if (missing.length) {
        const miss = el('div', 'category-missing');
        const label = el('span', 'category-missing-label');
        label.textContent = t('modMissingLabel') || '缺失:';
        miss.appendChild(label);
        for (const m of missing) {
          const link = el('button', 'missing-link');
          link.type = 'button';
          link.textContent = m.path;
          link.title = t('modMissingClickTip') || '点击打开文件夹';
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
