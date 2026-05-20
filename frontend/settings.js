'use strict';

(function () {
  const SOURCES = {
    env: { endpoint: '/api/env' },
    exportCmd: { endpoint: '/api/export-config' },
  };

  const SECTIONS = [
    {
      id: 'secrets',
      source: 'env',
      titleKey: 'settingsSectionSecrets',
      hintKey: 'envModalLeaveBlankHint',
      fields: [
        { key: 'deepseek_api_key', labelKey: 'envModalLabelDeepseekKey', sensitive: true, required: true },
        { key: 'HF_TOKEN', labelKey: 'envModalLabelHfToken', sensitive: true, required: true },
        { key: 'MILVUS_TOKEN', labelKey: 'envModalLabelMilvusToken', sensitive: true },
      ],
    },
    {
      id: 'embedding',
      source: 'env',
      titleKey: 'settingsSectionEmbedding',
      hintKey: null,
      fields: [
        { key: 'EMBEDDING_MODEL', labelKey: 'envModalLabelEmbeddingModel', required: true },
        { key: 'EMBEDDING_BATCH_SIZE', labelKey: 'envModalLabelEmbeddingBatch' },
        { key: 'MILVUS_URI', labelKey: 'envModalLabelMilvusUri' },
        { key: 'MILVUS_DB_NAME', labelKey: 'envModalLabelMilvusDb' },
        { key: 'DESC_COLLECTION_NAME', labelKey: 'envModalLabelDescCollection' },
      ],
    },
    {
      id: 'app',
      source: 'env',
      titleKey: 'settingsSectionApp',
      hintKey: null,
      fields: [
        { key: 'APP_HOST', labelKey: 'envModalLabelAppHost' },
        { key: 'APP_PORT', labelKey: 'envModalLabelAppPort' },
        { key: 'DESC_TOP_K', labelKey: 'envModalLabelDescTopK' },
        { key: 'CONTEXT_N', labelKey: 'envModalLabelContextN' },
        { key: 'CODE_CHARS', labelKey: 'envModalLabelCodeChars' },
      ],
    },
    {
      id: 'paths',
      source: 'env',
      titleKey: 'settingsSectionPaths',
      hintKey: 'settingsPathsHint',
      fields: [
        { key: 'GAME_ROOT', labelKey: 'envModalLabelGameRoot' },
        { key: 'EXPORT_TOOL_PATH', labelKey: 'envModalLabelExportToolPath' },
        { key: 'GODOT_TOOL_PATH', labelKey: 'envModalLabelGodotToolPath' },
      ],
    },
    {
      id: 'exportcmd',
      source: 'exportCmd',
      titleKey: 'settingsSectionExportCmd',
      hintKey: 'settingsExportCmdHint',
      fields: [
        { key: 'GODOT_PATH', labelKey: 'envModalLabelGodotPath' },
        { key: 'GODOT_EXPORT_PATH', labelKey: 'envModalLabelGodotExportPath' },
        { key: 'STS2_PATH', labelKey: 'envModalLabelSts2Path' },
      ],
    },
  ];

  const state = {
    activeId: SECTIONS[0].id,
    values: { env: {}, exportCmd: {} },
    pending: { env: {}, exportCmd: {} },
    sourceErrors: { env: null, exportCmd: null },
  };

  function tr(key, params) {
    return window.STS2_SHARED ? window.STS2_SHARED.t(key, params || {}) : key;
  }

  function activeSection() {
    return SECTIONS.find((s) => s.id === state.activeId) || SECTIONS[0];
  }

  function isFieldFilled(field, source, liveInputValue) {
    // Prefer the value currently typed in the visible input, then pending,
    // then the server-side value. For sensitive fields the GET response
    // only exposes a "present" flag.
    if (typeof liveInputValue === 'string' && liveInputValue.trim() !== '') return true;
    const pending = state.pending[source] || {};
    if (Object.prototype.hasOwnProperty.call(pending, field.key)) {
      return String(pending[field.key] || '').trim() !== '';
    }
    const value = (state.values[source] || {})[field.key];
    if (!value) return false;
    if (field.sensitive) return !!value.present;
    return typeof value.value === 'string' && value.value.trim() !== '';
  }

  function sectionHasMissingRequired(section) {
    return section.fields.some((f) => f.required && !isFieldFilled(f, section.source));
  }

  function setStatus(message, kind) {
    const el = document.getElementById('settingsStatus');
    if (!el) return;
    el.textContent = message || '';
    el.classList.remove('is-success', 'is-error');
    if (kind === 'success') el.classList.add('is-success');
    else if (kind === 'error') el.classList.add('is-error');
  }

  function captureCurrentInputs() {
    const inputs = document.querySelectorAll('#settingsFields input');
    inputs.forEach((input) => {
      const source = input.dataset.source || 'env';
      const sensitive = input.dataset.sensitive === '1';
      const bucket = state.pending[source];
      if (sensitive) {
        if (input.value !== '') bucket[input.name] = input.value;
        else delete bucket[input.name];
      } else {
        bucket[input.name] = input.value;
      }
    });
  }

  function fieldEl(field, source) {
    const div = document.createElement('div');
    div.className = 'settings-field';
    if (field.required) div.classList.add('is-required');

    const label = document.createElement('label');
    label.htmlFor = `setting-${field.key}`;

    const labelText = document.createElement('span');
    labelText.dataset.i18n = field.labelKey;
    labelText.textContent = tr(field.labelKey);
    label.appendChild(labelText);

    if (field.required) {
      const mark = document.createElement('span');
      mark.className = 'required-mark';
      mark.setAttribute('aria-hidden', 'true');
      mark.textContent = '*';
      label.appendChild(mark);
    }
    div.appendChild(label);

    const input = document.createElement('input');
    input.id = `setting-${field.key}`;
    input.name = field.key;
    input.dataset.source = source;
    input.dataset.sensitive = field.sensitive ? '1' : '0';
    input.autocomplete = 'off';
    input.spellcheck = false;
    if (field.required) input.required = true;

    const value = state.values[source][field.key];
    const pendingBucket = state.pending[source];
    if (field.sensitive) {
      input.type = 'password';
      if (value && value.masked && value.present) {
        input.placeholder = value.preview || '••••••••';
      }
      input.value = pendingBucket[field.key] || '';
    } else {
      input.type = 'text';
      if (Object.prototype.hasOwnProperty.call(pendingBucket, field.key)) {
        input.value = pendingBucket[field.key];
      } else {
        input.value = (value && typeof value.value === 'string') ? value.value : '';
      }
    }

    if (field.required) {
      const refreshMissing = () => {
        const filled = isFieldFilled(field, source, input.value);
        div.classList.toggle('is-missing', !filled);
      };
      refreshMissing();
      input.addEventListener('input', () => {
        refreshMissing();
        updateNavMissingState();
      });
    }

    div.appendChild(input);
    return div;
  }

  function updateNavMissingState() {
    const inputs = document.querySelectorAll('#settingsFields input');
    const liveBySource = {};
    inputs.forEach((input) => {
      const src = input.dataset.source || 'env';
      if (!liveBySource[src]) liveBySource[src] = {};
      liveBySource[src][input.name] = input.value;
    });
    document.querySelectorAll('.settings-nav-item').forEach((btn) => {
      const section = SECTIONS.find((s) => s.id === btn.dataset.section);
      if (!section) return;
      const missing = section.fields.some((f) => {
        if (!f.required) return false;
        const live = (liveBySource[section.source] || {})[f.key];
        return !isFieldFilled(f, section.source, live);
      });
      btn.classList.toggle('is-missing', missing);
    });
  }

  function renderNav() {
    const nav = document.getElementById('settingsNav');
    if (!nav) return;
    nav.innerHTML = '';
    SECTIONS.forEach((section) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      const classes = ['settings-nav-item'];
      if (section.id === state.activeId) classes.push('active');
      if (sectionHasMissingRequired(section)) classes.push('is-missing');
      btn.className = classes.join(' ');
      btn.dataset.section = section.id;
      btn.dataset.i18n = section.titleKey;
      btn.textContent = tr(section.titleKey);
      btn.addEventListener('click', () => {
        if (state.activeId === section.id) return;
        captureCurrentInputs();
        state.activeId = section.id;
        renderNav();
        renderPanel();
      });
      nav.appendChild(btn);
    });
  }

  function renderPanel() {
    const section = activeSection();
    const titleEl = document.getElementById('settingsPanelTitle');
    if (titleEl) {
      titleEl.dataset.i18n = section.titleKey;
      titleEl.textContent = tr(section.titleKey);
    }
    const hintEl = document.getElementById('settingsPanelHint');
    if (hintEl) {
      if (section.hintKey) {
        hintEl.dataset.i18n = section.hintKey;
        hintEl.textContent = tr(section.hintKey);
        hintEl.style.display = '';
      } else {
        hintEl.removeAttribute('data-i18n');
        hintEl.textContent = '';
        hintEl.style.display = 'none';
      }
    }
    const wrap = document.getElementById('settingsFields');
    if (!wrap) return;
    wrap.innerHTML = '';

    const err = state.sourceErrors[section.source];
    if (err) {
      const note = document.createElement('div');
      note.className = 'settings-source-error';
      note.textContent = err;
      wrap.appendChild(note);
    }
    section.fields.forEach((field) => wrap.appendChild(fieldEl(field, section.source)));

    if (window.STS2_SHARED && typeof window.STS2_SHARED.applyTranslations === 'function') {
      window.STS2_SHARED.applyTranslations();
    }
    updateNavMissingState();
  }

  async function loadSource(sourceKey) {
    const cfg = SOURCES[sourceKey];
    if (!cfg) return;
    try {
      const res = await fetch(cfg.endpoint);
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.ok === false) {
        state.values[sourceKey] = {};
        state.sourceErrors[sourceKey] = data.error || ('HTTP ' + res.status);
        return;
      }
      state.values[sourceKey] = data.values || {};
      state.sourceErrors[sourceKey] = null;
    } catch (err) {
      state.values[sourceKey] = {};
      state.sourceErrors[sourceKey] = err.message || String(err);
    }
  }

  async function load() {
    setStatus(tr('envModalLoading'));
    await Promise.all(Object.keys(SOURCES).map(loadSource));
    renderNav();
    renderPanel();
    // Surface the most user-actionable error first (env is required).
    if (state.sourceErrors.env) {
      setStatus(tr('envModalLoadFailed', { message: state.sourceErrors.env }), 'error');
    } else {
      setStatus('');
    }
  }

  async function saveSource(sourceKey) {
    const updates = state.pending[sourceKey];
    if (!updates || Object.keys(updates).length === 0) return { ok: true, skipped: true };
    const cfg = SOURCES[sourceKey];
    const res = await fetch(cfg.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || ('HTTP ' + res.status));
    }
    state.values[sourceKey] = data.values || {};
    state.pending[sourceKey] = {};
    return { ok: true };
  }

  async function save() {
    captureCurrentInputs();
    const btn = document.getElementById('settingsSaveBtn');
    if (btn) btn.disabled = true;
    setStatus(tr('envModalSaving'));
    try {
      for (const key of Object.keys(SOURCES)) {
        await saveSource(key);
      }
      renderPanel();
      setStatus(tr('envModalSaved'), 'success');
    } catch (err) {
      setStatus(tr('envModalSaveFailed', { message: err.message || String(err) }), 'error');
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function init() {
    const btn = document.getElementById('settingsSaveBtn');
    if (btn) btn.addEventListener('click', save);
    const navBtn = document.getElementById('settingsToggleBtn');
    if (navBtn) {
      navBtn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
      }, true);
    }
    window.addEventListener('sts2langchange', () => {
      captureCurrentInputs();
      renderNav();
      renderPanel();
    });
    load();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
